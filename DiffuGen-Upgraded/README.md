# DiffuGen-Upgraded

This repository contains the code for the DiffuGen-Upgraded project, a self-refining image generation agent.

## File Descriptions

### Root Directory

*   `docker-compose-agentic.yml`: Defines all the services (the `langgraph` agent, the `diffugen-mcp` service, Open WebUI, etc.) and how they are configured and connected.
*   `diffugen.py`: The code for the `diffugen-mcp` service, which is the "engine" that handles the actual image generation.
*   `character_manager.py`: This file contains the logic for managing character consistency.
*   `adetailer.py`: This file contains the implementation of the Adetailer for face and hand refinement.
*   `preprocessor.py`: This file contains the preprocessor for ControlNet.
*   `diffugen_openapi.py`: This file defines the OpenAPI specification for the `diffugen-mcp` service.

### `langgraph_agent` Directory

*   `agent_server.py`: The FastAPI server for the `langgraph` agent. This is the "front door" that receives requests from Open WebUI.
*   `workflow.py`: This is the heart of the `langgraph` agent. It defines the agent's logic and how it processes requests.
*   `mcp_client.py`: This is the client that the `langgraph` agent uses to communicate with the `diffugen-mcp` service.
*   `vram_manager.py`: This is the VRAM manager that orchestrates the loading and unloading of models to optimize VRAM usage.
