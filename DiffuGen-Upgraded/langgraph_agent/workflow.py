"""
LangGraph Workflow for Self-Refining Image Generation
Implements the full agentic loop with VRAM optimization
"""

import os
import logging
import json
from typing import TypedDict, Annotated, Sequence, Literal
import operator

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from vram_manager import VRAMOrchestrator, requires_llm_phase, requires_diffusion_phase
from mcp_client import DiffuGenMCPClient

logger = logging.getLogger(__name__)


# Define the state structure
class AgentState(TypedDict):
    """State that flows through the workflow"""
    # User input
    user_input: str
    task_type: Literal["generate", "refine"]
    init_image_base64: str | None
    parameters: dict

    # Workflow control
    enable_critique: bool
    max_iterations: int
    iteration_count: int

    # LLM interaction
    messages: Annotated[Sequence[HumanMessage | AIMessage | SystemMessage], operator.add]

    # Prompt refinement
    current_prompt: str
    final_prompt: str

    # Tool execution
    tool_call: dict | None
    tool_result: dict | None

    # Self-healing
    error_count: int
    last_error: str | None

    # Critique loop
    critique_history: list
    needs_refinement: bool

    # Final output
    final_image_base64: str | None
    image_path: str | None


class ImageGenerationWorkflow:
    """
    Production-grade agentic workflow for image generation

    Features:
    - Self-refining prompts
    - VRAM-optimized execution
    - Base64 injection
    - Self-healing on errors
    - Critique loop
    """

    def __init__(self, vram_manager: VRAMOrchestrator):
        self.vram_manager = vram_manager

        # Initialize LLM client (Ollama with OpenAI-compatible API)
        self.llm = ChatOpenAI(
            base_url=os.getenv("VLLM_API_BASE", "http://ollama:11434/v1"),
            api_key="EMPTY",
            model="qwen2.5:latest",
            temperature=0.7,
            max_tokens=2048
        )

        # Initialize MCP client
        self.mcp_client = DiffuGenMCPClient(
            base_url=os.getenv("DIFFUGEN_MCP_BASE", "http://diffugen-mcp:8080")
        )

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("analyze_request", self.analyze_request)
        workflow.add_node("refine_prompt", self.refine_prompt)
        workflow.add_node("generate_image", self.generate_image)
        workflow.add_node("critique_result", self.critique_result)
        workflow.add_node("handle_error", self.handle_error)

        # Define edges
        workflow.set_entry_point("analyze_request")

        # From analyze_request
        workflow.add_conditional_edges(
            "analyze_request",
            self._should_refine_prompt,
            {
                "refine": "refine_prompt",
                "generate": "generate_image"
            }
        )

        # From refine_prompt
        workflow.add_edge("refine_prompt", "generate_image")

        # From generate_image
        workflow.add_conditional_edges(
            "generate_image",
            self._check_generation_result,
            {
                "success": "critique_result",
                "error": "handle_error",
                "end": END
            }
        )

        # From critique_result
        workflow.add_conditional_edges(
            "critique_result",
            self._should_iterate,
            {
                "refine": "refine_prompt",
                "done": END
            }
        )

        # From handle_error
        workflow.add_conditional_edges(
            "handle_error",
            self._can_retry,
            {
                "retry": "generate_image",
                "fail": END
            }
        )

        return workflow.compile()

    @requires_llm_phase
    async def analyze_request(self, state: AgentState) -> AgentState:
        """
        Analyze user request and determine workflow path

        VRAM: LLM Phase (Qwen3 loaded)
        """
        logger.info("=== Node: Analyze Request ===")

        # Dynamically get LoRA models
        try:
            loras = await self.mcp_client.get_loras()
            lora_list = ", ".join(loras) if loras else "None"
        except Exception as e:
            logger.error(f"Failed to get LoRA models from MCP: {e}")
            lora_list = "None"

        system_prompt = f"""You are an expert AI assistant specializing in image generation.
Analyze the user's request to extract the prompt and any specified parameters.

AVAILABLE LoRA MODELS: {lora_list}

Extract the following parameters if mentioned:
- model: (e.g., "sd15", "sdxl")
- lora: (e.g., "rednose", "oia")
- width: (integer)
- height: (integer)
- steps: (integer)
- cfg_scale: (float)
- lora_strength: (float)
- negative_prompt: (string)

If the user mentions a LoRA model, you MUST set "needs_refinement" to true and include the LoRA in the "initial_prompt" using the format `<lora:MODEL_NAME:1.0>`.

If a word in the prompt matches a LoRA model name from the AVAILABLE LoRA MODELS list, you MUST treat it as a LoRA model and not as a literal description. Remove the word from the main prompt and place it in the `lora` parameter.

Respond in JSON format:
{{
    "needs_refinement": true/false,
    "reasoning": "explanation",
    "initial_prompt": "cleaned and structured prompt with LoRA if applicable",
    "parameters": {{
        "model": "...",
        "lora": "...",
        "width": "...",
        "height": "...",
        "steps": "...",
        "cfg_scale": "...",
        "lora_strength": "...",
        "negative_prompt": "..."
    }}
}}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User request: {state['user_input']}\nTask: {state['task_type']}")
        ]

        response = await self.llm.ainvoke(messages)

        # Parse response (simplified - in production use structured output)
        try:
            analysis = json.loads(response.content)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM response: {response.content}")
            analysis = {{"needs_refinement": False, "initial_prompt": state["user_input"], "parameters": {{}}}}

        state["current_prompt"] = analysis.get("initial_prompt", state["user_input"])
        state["needs_refinement"] = analysis.get("needs_refinement", False) and state["enable_critique"]
        
        # Update parameters from analysis
        if "parameters" in analysis:
            state["parameters"].update(analysis["parameters"])

        state["messages"] = messages + [response]

        logger.info(f"Analysis complete: needs_refinement={state['needs_refinement']}")

        return state

    @requires_llm_phase
    async def refine_prompt(self, state: AgentState) -> AgentState:
        """
        Refine the prompt using expert critique

        VRAM: LLM Phase (Qwen3 loaded)
        """
        logger.info("=== Node: Refine Prompt ===")

        # Get critique from previous iteration if exists
        last_critique = state["critique_history"][-1] if state["critique_history"] else None

        system_prompt = """You are an expert prompt engineer for image generation.
Given the current prompt and optional critique, create an improved version that will produce higher quality results.

Focus on:
- Specific, vivid descriptions
- Technical details (lighting, composition, style)
- Avoiding ambiguity
- Incorporating feedback from critique

Return ONLY the improved prompt text, nothing else."""

        messages = state["messages"] + [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Current prompt: {state['current_prompt']}

{'Critique from previous iteration: ' + last_critique if last_critique else 'First iteration - create the best prompt possible.'}

Improved prompt:""")
        ]

        response = await self.llm.ainvoke(messages)

        refined_prompt = response.content.strip()

        state["current_prompt"] = refined_prompt
        state["final_prompt"] = refined_prompt
        state["messages"] = messages + [response]

        logger.info(f"Prompt refined: {refined_prompt[:100]}...")

        return state

    @requires_diffusion_phase
    async def generate_image(self, state: AgentState) -> AgentState:
        """
        Execute image generation via DiffuGen MCP

        VRAM: Diffusion Phase (Stable Diffusion loaded, Qwen3 unloaded)

        THIS IS WHERE BASE64 INJECTION HAPPENS!
        """
        logger.info("=== Node: Generate Image ===")
        logger.info(f"Prompt: {state['current_prompt'][:100]}...")

        try:
            if state["task_type"] == "refine" and state["init_image_base64"]:
                # Image refinement with BASE64 INJECTION
                logger.info("Calling refine_image with base64 data")

                result = await self.mcp_client.refine_image(
                    prompt=state["current_prompt"],
                    init_image_base64=state["init_image_base64"],  # ← INJECTION!
                    model=state["parameters"].get("model", "sd15"),
                    strength=state["parameters"].get("strength", 0.5),
                    return_base64=True,
                    width=state["parameters"].get("width"),
                    height=state["parameters"].get("height"),
                    steps=state["parameters"].get("steps"),
                    cfg_scale=state["parameters"].get("cfg_scale"),
                    seed=state["parameters"].get("seed", -1),
                    sampling_method=state["parameters"].get("sampling_method"),
                    negative_prompt=state["parameters"].get("negative_prompt", ""),
                )
            else:
                # Text-to-image generation
                logger.info("Calling generate_image")

                result = await self.mcp_client.generate_image(
                    prompt=state["current_prompt"],
                    model=state["parameters"].get("model", "sd15"),
                    width=state["parameters"].get("width", 512),
                    height=state["parameters"].get("height", 512),
                    steps=state["parameters"].get("steps"),
                    cfg_scale=state["parameters"].get("cfg_scale"),
                    seed=state["parameters"].get("seed", -1),
                    sampling_method=state["parameters"].get("sampling_method"),
                    negative_prompt=state["parameters"].get("negative_prompt", ""),
                    lora=state["parameters"].get("lora"),
                    return_base64=True
                )

            state["tool_result"] = result

            if result.get("success"):
                state["final_image_base64"] = result.get("image_base64")
                state["image_path"] = result.get("image_path")
                state["error_count"] = 0  # Reset error count on success
                logger.info(f"Generation successful: {result.get('image_path')}")
            else:
                state["last_error"] = result.get("error", "Unknown error")
                state["error_count"] += 1
                logger.error(f"Generation failed: {state['last_error']}")

        except Exception as e:
            logger.error(f"Exception during generation: {e}", exc_info=True)
            state["last_error"] = str(e)
            state["error_count"] += 1
            state["tool_result"] = {"success": False, "error": str(e)}

        return state

    @requires_llm_phase
    async def critique_result(self, state: AgentState) -> AgentState:
        """
        Critique the generated image and decide if refinement is needed

        VRAM: LLM Phase (back to Qwen3)
        """
        logger.info("=== Node: Critique Result ===")

        if not state["enable_critique"] or state["iteration_count"] >= state["max_iterations"]:
            logger.info("Skipping critique (disabled or max iterations reached)")
            state["needs_refinement"] = False
            return state

        system_prompt = """You are an expert art director with vision capabilities.
Analyze the user's feedback and the generated image to provide targeted critique.

Consider:
- Does the image match the user's intent and feedback?
- Are there quality issues (e.g., artifacts, composition)?
- Could specific aspects be improved?

Respond in JSON format:
{
    "needs_refinement": true/false,
    "critique": "specific actionable feedback for the prompt",
    "suggested_parameters": {
        "lora_strength": "...",
        "steps": "...",
        "cfg_scale": "..."
    }
}"""

        # Create the message with the image
        message_content = [
            {"type": "text", "text": f"""Original request: {state['user_input']}
Current prompt: {state['current_prompt']}
User feedback: {state.get('user_feedback', 'No feedback provided.')}
Iteration: {state['iteration_count'] + 1}/{state['max_iterations']}

Provide your critique and suggested parameter changes:"""""}
        ]
        if state["final_image_base64"]:
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{state['final_image_base64']}"
                }
            })

        messages = state["messages"] + [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message_content)
        ]

        response = await self.llm.ainvoke(messages)

        logger.debug(f"Raw LLM response content: {response.content}")
        logger.debug(f"Type of LLM response content: {type(response.content)}")

        try:
            critique = json.loads(response.content)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from critique response: {response.content}")
            critique = {"needs_refinement": False, "critique": "Failed to parse critique.", "suggested_parameters": {}}

        state["critique_history"].append(critique.get("critique", ""))
        state["needs_refinement"] = critique.get("needs_refinement", False)
        
        # Update parameters from critique
        if "suggested_parameters" in critique:
            state["parameters"].update(critique["suggested_parameters"])

        state["iteration_count"] += 1
        state["messages"] = messages + [response]

        logger.info(f"Critique complete: needs_refinement={state['needs_refinement']}")

        return state

    @requires_llm_phase
    async def handle_error(self, state: AgentState) -> AgentState:
        """
        SELF-HEALING: Analyze error and attempt to fix

        VRAM: LLM Phase
        """
        logger.info("=== Node: Handle Error (Self-Healing) ===")

        system_prompt = """You are a debugging expert for image generation systems.
Analyze the error and determine how to fix it.

Common issues:
- Missing required parameters
- Invalid parameter values
- Base64 encoding problems
- Model compatibility issues

Respond in JSON:
{
    "diagnosis": "what went wrong",
    "fix_strategy": "how to fix it",
    "can_retry": true/false,
    "corrected_parameters": {...}
}"""

        messages = state["messages"] + [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Error occurred: {state['last_error']}
Tool call: {state.get('tool_call')}
Error count: {state['error_count']}

Diagnose and suggest fix:""")
        ]

        response = await self.llm.ainvoke(messages)

        try:
            diagnosis = json.loads(response.content)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from error diagnosis response: {response.content}")
            diagnosis = {"can_retry": False, "diagnosis": "Failed to parse diagnosis."}

        # Apply corrections if suggested
        if diagnosis.get("corrected_parameters"):
            state["parameters"].update(diagnosis["corrected_parameters"])

        state["messages"] = messages + [response]

        logger.info(f"Error diagnosis: {diagnosis.get('diagnosis')}")

        return state

    # Conditional edge functions
    def _should_refine_prompt(self, state: AgentState) -> str:
        """Decide if prompt needs refinement before generation"""
        return "refine" if state.get("needs_refinement", False) else "generate"

    def _check_generation_result(self, state: AgentState) -> str:
        """Check if generation succeeded"""
        if state.get("tool_result", {}).get("success"):
            if state["enable_critique"]:
                return "success"  # Go to critique
            else:
                return "end"  # Skip critique, done
        else:
            return "error"  # Go to error handler

    def _should_iterate(self, state: AgentState) -> str:
        """Decide if another refinement iteration is needed"""
        if state.get("needs_refinement", False) and state["iteration_count"] < state["max_iterations"]:
            return "refine"
        return "done"

    def _can_retry(self, state: AgentState) -> str:
        """Decide if we should retry after error"""
        max_retries = int(os.getenv("MAX_RETRIES", "3"))
        if state["error_count"] < max_retries:
            return "retry"
        return "fail"

    async def run(self, initial_state: dict) -> dict:
        """Execute the workflow"""
        # Initialize state
        state = AgentState(
            user_input=initial_state["user_input"],
            task_type=initial_state["task_type"],
            init_image_base64=initial_state.get("init_image_base64"),
            parameters=initial_state.get("parameters", {}),
            enable_critique=initial_state.get("enable_critique", True),
            max_iterations=initial_state.get("max_iterations", 2),
            iteration_count=0,
            messages=[],
            current_prompt=initial_state["user_input"],
            final_prompt="",
            tool_call=None,
            tool_result=None,
            error_count=0,
            last_error=None,
            critique_history=[],
            needs_refinement=False,
            final_image_base64=None,
            image_path=None
        )

        # Run the graph
        final_state = await self.graph.ainvoke(state)

        return final_state


def create_image_generation_workflow(vram_manager: VRAMOrchestrator) -> ImageGenerationWorkflow:
    """Factory function to create workflow"""
    return ImageGenerationWorkflow(vram_manager)
