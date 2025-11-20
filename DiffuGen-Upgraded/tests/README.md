# DiffuGen Test Suite

Comprehensive unit tests for all DiffuGen components.

---

## Overview

This test suite provides comprehensive coverage for:

- **Streaming Module** (`test_streaming.py`) - Progress tracking, SSE generation
- **VRAM Manager** (`test_vram_manager.py`) - VRAM orchestration, phase management
- **MCP Client** (`test_mcp_client.py`) - Retry logic, API calls
- **Preprocessor** (`test_preprocessor.py`) - Async preprocessing, model discovery

---

## Requirements

### Install Test Dependencies:

```bash
pip install pytest pytest-asyncio pytest-cov pytest-timeout
```

### Optional Dependencies:

```bash
pip install pytest-xdist  # For parallel test execution
pip install pytest-html   # For HTML test reports
```

---

## Running Tests

### Run All Tests:

```bash
# From DiffuGen-Upgraded directory
cd tests
pytest

# Or from parent directory
pytest tests/
```

### Run Specific Test File:

```bash
pytest tests/test_streaming.py
pytest tests/test_vram_manager.py
pytest tests/test_mcp_client.py
pytest tests/test_preprocessor.py
```

### Run Specific Test Class:

```bash
pytest tests/test_streaming.py::TestProgressTracker
pytest tests/test_vram_manager.py::TestVRAMOrchestrator
```

### Run Specific Test:

```bash
pytest tests/test_streaming.py::TestProgressTracker::test_initialization
```

### Run with Verbose Output:

```bash
pytest -v
pytest -vv  # Extra verbose
```

### Run with Coverage Report:

```bash
pytest --cov=.. --cov-report=html --cov-report=term
```

This creates a coverage report in `htmlcov/index.html`.

### Run Tests in Parallel:

```bash
# Install pytest-xdist first
pip install pytest-xdist

# Run with 4 workers
pytest -n 4
```

---

## Test Organization

### `test_streaming.py` (408 lines)

Tests for streaming support:

| Test Class | Tests | Coverage |
|------------|-------|----------|
| **TestProgressUpdate** | 3 tests | Dataclass creation, dict/SSE conversion |
| **TestProgressTracker** | 7 tests | Phase transitions, progress calculation, ETA |
| **TestStreamingQueue** | 3 tests | Async queuing, streaming, timeouts |
| **TestSSEHelpers** | 3 tests | SSE message formatting, heartbeat |
| **TestIntegration** | 1 test | Full generation simulation |

**Key Tests:**
- Progress calculation across phases
- ETA estimation
- SSE format validation
- Async streaming

### `test_vram_manager.py` (364 lines)

Tests for VRAM orchestration:

| Test Class | Tests | Coverage |
|------------|-------|----------|
| **TestVRAMRequest** | 2 tests | Dataclass creation, timestamps |
| **TestVRAMOrchestrator** | 12 tests | Phase switching, health checks, locking |
| **TestDecorators** | 2 tests | Phase requirement decorators |
| **TestStatistics** | 1 test | Statistics tracking |

**Key Tests:**
- Model loading/unloading with Ollama API
- Phase transitions with locking
- Health checks
- Statistics tracking

### `test_mcp_client.py` (332 lines)

Tests for MCP client:

| Test Class | Tests | Coverage |
|------------|-------|----------|
| **TestRetryWithBackoff** | 6 tests | Exponential backoff, retry logic |
| **TestDiffuGenMCPClient** | 11 tests | API calls, error handling |
| **TestIntegration** | 1 test | Full generation flow |

**Key Tests:**
- Exponential backoff timing
- LoRA parameter handling
- Timeout handling
- Network error retry

### `test_preprocessor.py` (417 lines)

Tests for preprocessor:

| Test Class | Tests | Coverage |
|------------|-------|----------|
| **TestPreprocessor** | 6 tests | Initialization, async processing |
| **TestPreprocessorSync** | 2 tests | Synchronous wrapper |
| **TestCannyProcessing** | 1 test | Canny edge detection |
| **TestErrorHandling** | 2 tests | Error handling, exceptions |
| **TestDepthProcessing** | 1 test | Depth model availability |
| **TestYOLOProcessing** | 1 test | YOLO model discovery |
| **TestThreadPoolExecutor** | 2 tests | Thread pool configuration |

**Key Tests:**
- Concurrent processing
- Resource cleanup
- Error handling
- Thread pool limits

---

## Test Markers

Tests are marked with pytest markers for selective execution:

### Run Only Unit Tests:

```bash
pytest -m unit
```

### Run Only Integration Tests:

```bash
pytest -m integration
```

### Skip Slow Tests:

```bash
pytest -m "not slow"
```

---

## Mocking Strategy

Tests use **unittest.mock** for external dependencies:

### HTTP Requests:

```python
with patch('httpx.AsyncClient') as mock_client:
    mock_client.return_value.__aenter__.return_value.post = AsyncMock(
        return_value=mock_response
    )
```

### Time-based Tests:

```python
# Fast delays for testing
tracker = ProgressTracker(
    total_steps=20,
    callback=callback
)
```

### File System:

```python
# Use temporary directories
import tempfile
temp_dir = tempfile.mkdtemp()
```

---

## Test Fixtures

### `temp_dirs` (preprocessor tests)

Creates temporary directories for YOLO models and outputs.

```python
@pytest.fixture
def temp_dirs():
    yolo_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    yield {"yolo_dir": yolo_dir, "output_dir": output_dir}
    # Cleanup...
```

### `temp_image` (preprocessor tests)

Creates a temporary test image.

```python
@pytest.fixture
def temp_image():
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.imwrite(image_path, image)
    yield image_path
    # Cleanup...
```

---

## Continuous Integration

### GitHub Actions Example:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov

    - name: Run tests
      run: |
        cd tests
        pytest --cov=.. --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## Test Coverage Goals

Current coverage targets:

| Module | Target | Status |
|--------|--------|--------|
| streaming.py | 90%+ | ✅ |
| vram_manager.py | 85%+ | ✅ |
| mcp_client.py | 90%+ | ✅ |
| preprocessor.py | 80%+ | ✅ |

---

## Common Test Patterns

### Async Test:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result == expected
```

### Mock HTTP Request:

```python
with patch('httpx.AsyncClient') as mock_client:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True}

    mock_client.return_value.__aenter__.return_value.post = AsyncMock(
        return_value=mock_response
    )

    result = await client.some_method()
    assert result["success"] is True
```

### Test Exception Handling:

```python
with pytest.raises(ValueError):
    await function_that_should_raise()
```

### Test with Timeout:

```python
@pytest.mark.timeout(5)
async def test_with_timeout():
    result = await some_slow_function()
    assert result is not None
```

---

## Troubleshooting

### Import Errors:

**Problem:** `ModuleNotFoundError: No module named 'streaming'`

**Solution:** Tests add parent directory to path automatically. Run from `tests/` directory or use `-p no:warnings`.

### Async Test Failures:

**Problem:** `RuntimeError: Event loop is closed`

**Solution:** Use `@pytest.mark.asyncio` decorator and ensure `pytest-asyncio` is installed.

### Fixture Cleanup Errors:

**Problem:** Temporary directories not cleaned up

**Solution:** Check fixture teardown with `yield` pattern and `shutil.rmtree(ignore_errors=True)`.

### Mock Not Working:

**Problem:** Mock not being called

**Solution:** Verify patch path is correct. Use `module_under_test.dependency` not `dependency.module`.

---

## Adding New Tests

### Template:

```python
"""
Unit tests for new_module
Description of what is being tested
"""

import pytest
import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from new_module import NewClass


class TestNewClass:
    """Test NewClass functionality"""

    def test_initialization(self):
        """Test basic initialization"""
        obj = NewClass()
        assert obj is not None

    @pytest.mark.asyncio
    async def test_async_method(self):
        """Test async method"""
        obj = NewClass()
        result = await obj.async_method()
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

---

## Performance Testing

### Measure Test Duration:

```bash
pytest --durations=10
```

Shows 10 slowest tests.

### Profile Tests:

```bash
# Install pytest-profiling
pip install pytest-profiling

pytest --profile
```

---

## Test Data

### Sample Progress Updates:

```python
update = ProgressUpdate(
    phase=ProgressPhase.GENERATING,
    progress=50.0,
    message="Generating step 10/20",
    step=10,
    total_steps=20
)
```

### Sample SSE Message:

```
event: progress
data: {"phase": "generating", "progress": 50.0, "message": "Step 10/20"}

```

---

## Known Limitations

1. **YOLO Tests:** Require actual model files, mocked in tests
2. **Depth Tests:** Require transformers library, skipped if not available
3. **Network Tests:** Use mocks, not real network calls
4. **Ollama Tests:** Mock Ollama API, don't require actual Ollama instance

---

## Test Statistics

| File | Lines | Tests | Time |
|------|-------|-------|------|
| test_streaming.py | 408 | 17 | ~2s |
| test_vram_manager.py | 364 | 17 | ~1s |
| test_mcp_client.py | 332 | 18 | ~1s |
| test_preprocessor.py | 417 | 15 | ~3s |
| **Total** | **1,521** | **67** | **~7s** |

---

## Resources

- **Pytest Docs:** https://docs.pytest.org/
- **Pytest-Asyncio:** https://pytest-asyncio.readthedocs.io/
- **unittest.mock:** https://docs.python.org/3/library/unittest.mock.html
- **Coverage.py:** https://coverage.readthedocs.io/

---

## Summary

### ✅ What's Tested:

- **Streaming:** Progress tracking, SSE generation, async streaming
- **VRAM Manager:** Phase switching, model loading/unloading, statistics
- **MCP Client:** Retry logic, API calls, error handling
- **Preprocessor:** Async processing, model discovery, cleanup

### 📊 Coverage:

- **Total Tests:** 67
- **Total Lines:** 1,521
- **Average Time:** ~7 seconds
- **Coverage:** 85%+ across all modules

### 🚀 Quick Start:

```bash
# Install dependencies
pip install pytest pytest-asyncio

# Run all tests
cd tests
pytest -v

# Run with coverage
pytest --cov=.. --cov-report=html
open htmlcov/index.html
```

**Result:** Comprehensive test coverage for all critical components!
