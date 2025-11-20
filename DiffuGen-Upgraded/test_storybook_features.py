"""
Comprehensive Test Suite for Storybook Features
Tests all 4 major enhancements + hybrid LLM router
"""

import asyncio
import sys
from pathlib import Path

print("="*70)
print("DiffuGen Storybook Features - Comprehensive Test Suite")
print("="*70)

# Track test results
tests_passed = 0
tests_failed = 0
tests_skipped = 0

def test_result(name, passed, reason=""):
    """Record test result"""
    global tests_passed, tests_failed, tests_skipped

    if passed is None:
        print(f"⊘ SKIP: {name} - {reason}")
        tests_skipped += 1
    elif passed:
        print(f"✓ PASS: {name}")
        tests_passed += 1
    else:
        print(f"✗ FAIL: {name} - {reason}")
        tests_failed += 1


# ============================================================================
# Test 1: Module Imports
# ============================================================================

print("\n" + "="*70)
print("TEST SUITE 1: Module Imports")
print("="*70)

try:
    import lora_training
    test_result("Import lora_training", True)
except Exception as e:
    test_result("Import lora_training", False, str(e))

try:
    import style_locking
    test_result("Import style_locking", True)
except Exception as e:
    test_result("Import style_locking", False, str(e))

try:
    import batch_generation
    test_result("Import batch_generation", True)
except Exception as e:
    test_result("Import batch_generation", False, str(e))

try:
    import character_relationships
    test_result("Import character_relationships", True)
except Exception as e:
    test_result("Import character_relationships", False, str(e))

try:
    import llm_router
    test_result("Import llm_router", True)
except Exception as e:
    test_result("Import llm_router", False, str(e))

try:
    import intelligent_workflow
    test_result("Import intelligent_workflow", True)
except Exception as e:
    test_result("Import intelligent_workflow", False, str(e))


# ============================================================================
# Test 2: Data Models
# ============================================================================

print("\n" + "="*70)
print("TEST SUITE 2: Data Models")
print("="*70)

try:
    from lora_training import LoRATrainingConfig, TrainingProgress
    config = LoRATrainingConfig(
        character_name="TestChar",
        training_images_dir="test",
        output_dir="test_output"
    )
    assert config.character_name == "TestChar"
    assert config.epochs == 10  # default
    test_result("LoRATrainingConfig creation", True)
except Exception as e:
    test_result("LoRATrainingConfig creation", False, str(e))

try:
    from style_locking import ArtStyle
    style = ArtStyle(
        name="test_style",
        description="test description",
        technique="watercolor",
        color_palette="pastel",
        mood="calm",
        detail_level="moderate"
    )
    assert style.name == "test_style"
    assert style.cfg_scale == 7.5  # default
    test_result("ArtStyle creation", True)
except Exception as e:
    test_result("ArtStyle creation", False, str(e))

try:
    from batch_generation import SceneDefinition, BatchJob
    scene = SceneDefinition(
        id="scene_001",
        description="test scene"
    )
    assert scene.status == "pending"
    test_result("SceneDefinition creation", True)
except Exception as e:
    test_result("SceneDefinition creation", False, str(e))

try:
    from character_relationships import Relationship, RelationType
    rel = Relationship(
        character_a="Alice",
        character_b="Bob",
        relationship_type=RelationType.FRIEND
    )
    assert rel.character_a == "Alice"
    test_result("Relationship creation", True)
except Exception as e:
    test_result("Relationship creation", False, str(e))


# ============================================================================
# Test 3: Style Presets
# ============================================================================

print("\n" + "="*70)
print("TEST SUITE 3: Style System")
print("="*70)

try:
    from style_locking import PRESET_STYLES, StyleLibrary

    # Check presets exist
    assert "watercolor_soft" in PRESET_STYLES
    assert "digital_vibrant" in PRESET_STYLES
    assert "pencil_sketch" in PRESET_STYLES
    assert "cartoon_bold" in PRESET_STYLES
    assert "storybook_classic" in PRESET_STYLES
    test_result("All 5 preset styles exist", True)

    # Test style library
    library = StyleLibrary()
    assert len(library.styles) >= 5  # At least the presets
    test_result("StyleLibrary initialization", True)

    # Test getting a style
    watercolor = library.get_style("watercolor_soft")
    assert watercolor is not None
    assert watercolor.name == "Watercolor Soft"
    test_result("Get preset style", True)

except Exception as e:
    test_result("Style system tests", False, str(e))


# ============================================================================
# Test 4: Relationship Graph
# ============================================================================

print("\n" + "="*70)
print("TEST SUITE 4: Character Relationships")
print("="*70)

try:
    from character_relationships import RelationshipGraph, RelationType

    graph = RelationshipGraph()
    test_result("RelationshipGraph initialization", True)

    # Add a relationship
    rel = graph.add_relationship(
        "Spark",
        "Whiskers",
        RelationType.FRIEND,
        description="best friends"
    )
    assert rel.character_a == "Spark"
    test_result("Add relationship", True)

    # Query relationships
    spark_rels = graph.get_relationships("Spark")
    assert len(spark_rels) >= 1
    test_result("Query relationships", True)

    # Check bidirectional
    whiskers_rels = graph.get_relationships("Whiskers")
    assert len(whiskers_rels) >= 1
    test_result("Bidirectional relationships", True)

except Exception as e:
    test_result("Relationship graph tests", False, str(e))


# ============================================================================
# Test 5: Routing Logic
# ============================================================================

print("\n" + "="*70)
print("TEST SUITE 5: LLM Router (without API calls)")
print("="*70)

try:
    from llm_router import MessageRouter, LLMType, UsageStats

    # Test usage stats
    stats = UsageStats()
    stats.record_gemini_call()
    stats.record_qwen_vl_call()

    result = stats.get_stats()
    assert result["gemini_calls_today"] == 1
    assert result["qwen_vl_calls_today"] == 1
    assert result["gemini_calls_remaining"] == 1499
    test_result("UsageStats tracking", True)

except Exception as e:
    test_result("Router stats tests", False, str(e))

try:
    from llm_router import GeminiAnalyzer, QwenVLAnalyzer

    # Test Gemini (should be disabled without API key)
    gemini = GeminiAnalyzer()
    assert gemini.enabled == False  # No API key
    test_result("GeminiAnalyzer graceful degradation", True)

    # Test Qwen-VL initialization
    qwen_vl = QwenVLAnalyzer()
    # Just check it doesn't crash
    test_result("QwenVLAnalyzer initialization", True)

except Exception as e:
    test_result("Analyzer initialization tests", False, str(e))


# ============================================================================
# Test 6: Helper Functions
# ============================================================================

print("\n" + "="*70)
print("TEST SUITE 6: Helper Functions")
print("="*70)

try:
    from batch_generation import parse_batch_request

    # Test pattern 1
    scenes = parse_batch_request("Generate scenes: castle, forest, beach")
    assert scenes is not None
    assert len(scenes) == 3
    assert "castle" in scenes
    test_result("Parse batch request (pattern 1)", True)

    # Test pattern 2
    scenes = parse_batch_request("Create 5 scenes: morning, noon, afternoon, evening, night")
    assert scenes is not None
    assert len(scenes) == 5
    test_result("Parse batch request (pattern 2)", True)

except Exception as e:
    test_result("Batch parsing tests", False, str(e))

try:
    from character_relationships import parse_relationship_request

    # Test pattern 1
    result = parse_relationship_request("Spark's friend is Whiskers")
    assert result is not None
    assert result[0] == "Spark"
    assert result[1] == "Whiskers"
    assert result[2] == "friend"
    test_result("Parse relationship (pattern 1)", True)

    # Test pattern 2
    result = parse_relationship_request("Luna is Spark's sister")
    assert result is not None
    assert "Luna" in result
    assert "Spark" in result
    test_result("Parse relationship (pattern 2)", True)

except Exception as e:
    test_result("Relationship parsing tests", False, str(e))

try:
    from lora_training import create_lora_prompt_tag, extract_lora_from_prompt

    # Test tag creation
    tag = create_lora_prompt_tag("spark", 1.0)
    assert tag == "<lora:spark:1.0>"
    test_result("Create LoRA prompt tag", True)

    # Test extraction
    prompt = "dragon in castle <lora:spark:0.8> colorful"
    cleaned, loras = extract_lora_from_prompt(prompt)
    assert len(loras) == 1
    assert loras[0][0] == "spark"
    assert loras[0][1] == 0.8
    assert "<lora:" not in cleaned
    test_result("Extract LoRA from prompt", True)

except Exception as e:
    test_result("LoRA helper functions", False, str(e))


# ============================================================================
# Test 7: Integration Points
# ============================================================================

print("\n" + "="*70)
print("TEST SUITE 7: Integration Points")
print("="*70)

try:
    from character_consistency import Character

    # Test Character with LoRA fields
    char = Character(
        name="TestChar",
        description="test",
        reference_image="test.png",
        seed=12345
    )
    assert char.has_lora == False  # default
    assert char.lora_path is None
    assert char.lora_weight == 1.0
    test_result("Character with LoRA fields", True)

    # Test setting LoRA
    char.has_lora = True
    char.lora_path = "loras/testchar.safetensors"
    assert char.has_lora == True
    test_result("Set Character LoRA fields", True)

except Exception as e:
    test_result("Character-LoRA integration", False, str(e))


# ============================================================================
# Final Summary
# ============================================================================

print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print(f"✓ Passed:  {tests_passed}")
print(f"✗ Failed:  {tests_failed}")
print(f"⊘ Skipped: {tests_skipped}")
print(f"Total:     {tests_passed + tests_failed + tests_skipped}")
print("="*70)

if tests_failed > 0:
    print("\n⚠️  Some tests failed. Please review the errors above.")
    print("Common issues:")
    print("  - Missing dependencies (run: pip install -r requirements-storybook.txt)")
    print("  - Missing API keys (check .env file)")
    sys.exit(1)
else:
    print("\n✓ All tests passed! System is ready.")
    sys.exit(0)
