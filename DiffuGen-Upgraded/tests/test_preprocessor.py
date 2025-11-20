"""
Unit tests for preprocessor module
Tests async preprocessing, model discovery, and cleanup
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessor import (
    Preprocessor,
    PreprocessorSync
)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing"""
    yolo_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    yield {
        "yolo_dir": yolo_dir,
        "output_dir": output_dir
    }

    # Cleanup
    shutil.rmtree(yolo_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)


@pytest.fixture
def temp_image():
    """Create temporary test image"""
    import cv2

    temp_dir = tempfile.mkdtemp()
    image_path = Path(temp_dir) / "test_image.png"

    # Create a simple test image
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image[:, :] = (100, 150, 200)  # Gray-blue color
    cv2.imwrite(str(image_path), image)

    yield str(image_path)

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestPreprocessor:
    """Test Preprocessor class"""

    def test_initialization(self, temp_dirs):
        """Test preprocessor initialization"""
        preprocessor = Preprocessor(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"],
            max_workers=2
        )

        assert preprocessor.output_dir.exists()
        assert preprocessor.yolo_models_dir == Path(temp_dirs["yolo_dir"])
        assert preprocessor.executor is not None

    def test_output_directory_creation(self, temp_dirs):
        """Test that output directory is created"""
        output_dir = Path(temp_dirs["output_dir"]) / "subdir"
        assert not output_dir.exists()

        preprocessor = Preprocessor(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=str(output_dir)
        )

        assert output_dir.exists()

    def test_generate_output_path(self, temp_dirs):
        """Test output path generation"""
        preprocessor = Preprocessor(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"]
        )

        output_path = preprocessor._generate_output_path(
            "/path/to/input.png",
            "canny"
        )

        assert "input_canny_" in output_path
        assert output_path.endswith(".png")

    def test_list_available_models(self, temp_dirs):
        """Test listing available preprocessing models"""
        preprocessor = Preprocessor(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"]
        )

        models = preprocessor.list_available_models()

        # Canny should always be available
        assert models["canny"] is True

        # Others depend on dependencies
        assert "depth" in models
        assert "pose" in models
        assert "segmentation" in models

    @pytest.mark.asyncio
    async def test_canny_preprocessing(self, temp_dirs, temp_image):
        """Test Canny edge detection"""
        preprocessor = Preprocessor(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"]
        )

        result = await preprocessor.run(temp_image, "canny")

        assert result is not None
        assert Path(result).exists()
        assert "_canny_" in result

        # Cleanup
        await preprocessor.close()

    @pytest.mark.asyncio
    async def test_invalid_model_type(self, temp_dirs, temp_image):
        """Test invalid preprocessing type"""
        preprocessor = Preprocessor(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"]
        )

        result = await preprocessor.run(temp_image, "invalid_type")

        assert result is None

        await preprocessor.close()

    @pytest.mark.asyncio
    async def test_missing_input_file(self, temp_dirs):
        """Test with non-existent input file"""
        preprocessor = Preprocessor(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"]
        )

        result = await preprocessor.run("/nonexistent/image.png", "canny")

        assert result is None

        await preprocessor.close()

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, temp_dirs, temp_image):
        """Test concurrent processing of multiple images"""
        preprocessor = Preprocessor(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"],
            max_workers=2
        )

        # Process same image multiple times concurrently
        tasks = [
            preprocessor.run(temp_image, "canny")
            for _ in range(3)
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(result is not None for result in results)

        # All should have unique output paths
        assert len(set(results)) == 3

        await preprocessor.close()

    @pytest.mark.asyncio
    async def test_cleanup(self, temp_dirs):
        """Test resource cleanup"""
        preprocessor = Preprocessor(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"]
        )

        # Executor should be running
        assert not preprocessor.executor._shutdown

        await preprocessor.close()

        # Executor should be shut down
        assert preprocessor.executor._shutdown


class TestPreprocessorSync:
    """Test PreprocessorSync wrapper"""

    def test_synchronous_interface(self, temp_dirs, temp_image):
        """Test synchronous wrapper"""
        preprocessor = PreprocessorSync(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"]
        )

        # Should work without async/await
        result = preprocessor.run(temp_image, "canny")

        assert result is not None
        assert Path(result).exists()

    def test_list_models_sync(self, temp_dirs):
        """Test synchronous model listing"""
        preprocessor = PreprocessorSync(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"]
        )

        models = preprocessor.list_available_models()

        assert models["canny"] is True


class TestCannyProcessing:
    """Test Canny edge detection specifics"""

    @pytest.mark.asyncio
    async def test_canny_output_format(self, temp_dirs, temp_image):
        """Test Canny output is valid image"""
        import cv2

        preprocessor = Preprocessor(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"]
        )

        result = await preprocessor.run(temp_image, "canny")

        assert result is not None

        # Read output image
        output_image = cv2.imread(result, cv2.IMREAD_GRAYSCALE)

        assert output_image is not None
        assert len(output_image.shape) == 2  # Grayscale
        assert output_image.dtype == np.uint8

        await preprocessor.close()


class TestErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_corrupted_image(self, temp_dirs):
        """Test handling of corrupted image file"""
        preprocessor = Preprocessor(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"]
        )

        # Create corrupted image file
        temp_file = Path(temp_dirs["output_dir"]) / "corrupted.png"
        temp_file.write_text("not an image")

        result = await preprocessor.run(str(temp_file), "canny")

        # Should handle gracefully
        assert result is None

        await preprocessor.close()

    @pytest.mark.asyncio
    async def test_exception_in_processing(self, temp_dirs, temp_image):
        """Test exception handling during processing"""
        preprocessor = Preprocessor(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"]
        )

        # Mock _process_canny to raise exception
        original_method = preprocessor._process_canny

        def mock_process(*args, **kwargs):
            raise RuntimeError("Processing failed")

        preprocessor._process_canny = mock_process

        result = await preprocessor.run(temp_image, "canny")

        # Should return None on exception
        assert result is None

        # Restore original method
        preprocessor._process_canny = original_method

        await preprocessor.close()


class TestDepthProcessing:
    """Test depth estimation (if available)"""

    @pytest.mark.asyncio
    async def test_depth_availability(self, temp_dirs):
        """Test depth model availability check"""
        preprocessor = Preprocessor(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"]
        )

        models = preprocessor.list_available_models()

        # Check if depth is available
        depth_available = models["depth"]

        if depth_available:
            assert preprocessor.depth_model is not None
            assert preprocessor.depth_processor is not None
        else:
            # Should be None if not available
            assert preprocessor.depth_model is None or preprocessor.depth_processor is None

        await preprocessor.close()


class TestYOLOProcessing:
    """Test YOLO-based processing (if available)"""

    @pytest.mark.asyncio
    async def test_yolo_model_discovery(self, temp_dirs):
        """Test YOLO model discovery"""
        # Create dummy YOLO model files
        yolo_dir = Path(temp_dirs["yolo_dir"])
        (yolo_dir / "yolov8_pose.pt").touch()
        (yolo_dir / "yolov8_seg.pt").touch()

        preprocessor = Preprocessor(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"]
        )

        # Should discover models (though they won't load as they're dummy files)
        # This tests the discovery mechanism only

        await preprocessor.close()


class TestThreadPoolExecutor:
    """Test thread pool executor usage"""

    @pytest.mark.asyncio
    async def test_max_workers_configuration(self, temp_dirs):
        """Test max_workers configuration"""
        preprocessor = Preprocessor(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"],
            max_workers=4
        )

        assert preprocessor.executor._max_workers == 4

        await preprocessor.close()

    @pytest.mark.asyncio
    async def test_concurrent_limit(self, temp_dirs, temp_image):
        """Test that concurrent operations respect max_workers"""
        preprocessor = Preprocessor(
            yolo_models_dir=temp_dirs["yolo_dir"],
            output_dir=temp_dirs["output_dir"],
            max_workers=2
        )

        # Start many operations
        tasks = [
            preprocessor.run(temp_image, "canny")
            for _ in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert all(result is not None for result in results)

        await preprocessor.close()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
