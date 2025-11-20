"""
Unit tests for streaming module
Tests ProgressTracker, StreamingQueue, and SSE generation
"""

import pytest
import asyncio
import json
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming import (
    ProgressTracker,
    ProgressPhase,
    ProgressUpdate,
    StreamingQueue,
    sse_generator,
    create_sse_message,
    create_sse_heartbeat
)


class TestProgressUpdate:
    """Test ProgressUpdate dataclass"""

    def test_creation(self):
        """Test basic creation"""
        update = ProgressUpdate(
            phase=ProgressPhase.GENERATING,
            progress=50.0,
            message="Test message"
        )

        assert update.phase == ProgressPhase.GENERATING
        assert update.progress == 50.0
        assert update.message == "Test message"
        assert update.timestamp is not None

    def test_to_dict(self):
        """Test conversion to dictionary"""
        update = ProgressUpdate(
            phase=ProgressPhase.GENERATING,
            progress=75.5,
            message="Step 15/20",
            step=15,
            total_steps=20,
            eta_seconds=5.5
        )

        data = update.to_dict()

        assert data["phase"] == "generating"
        assert data["progress"] == 75.5
        assert data["message"] == "Step 15/20"
        assert data["step"] == 15
        assert data["total_steps"] == 20
        assert data["eta_seconds"] == 5.5
        assert "timestamp" in data

    def test_to_sse(self):
        """Test SSE format conversion"""
        update = ProgressUpdate(
            phase=ProgressPhase.COMPLETE,
            progress=100.0,
            message="Done"
        )

        sse = update.to_sse()

        assert sse.startswith("event: progress\n")
        assert "data:" in sse
        assert sse.endswith("\n\n")

        # Parse JSON from SSE
        data_line = [line for line in sse.split("\n") if line.startswith("data:")][0]
        json_str = data_line[5:].strip()
        data = json.loads(json_str)

        assert data["phase"] == "complete"
        assert data["progress"] == 100.0


class TestProgressTracker:
    """Test ProgressTracker functionality"""

    def test_initialization(self):
        """Test tracker initialization"""
        tracker = ProgressTracker(total_steps=20)

        assert tracker.total_steps == 20
        assert tracker.current_step == 0
        assert tracker.phase == ProgressPhase.INITIALIZING

    def test_phase_transition(self):
        """Test phase transitions"""
        updates = []

        def collect_update(update):
            updates.append(update)

        tracker = ProgressTracker(total_steps=20, callback=collect_update)

        # Transition through phases
        tracker.update_phase(ProgressPhase.LOADING_MODEL, "Loading...")
        assert tracker.phase == ProgressPhase.LOADING_MODEL
        assert len(updates) == 1
        assert updates[0].phase == ProgressPhase.LOADING_MODEL

        tracker.update_phase(ProgressPhase.GENERATING, "Generating...")
        assert len(updates) == 2
        assert updates[1].phase == ProgressPhase.GENERATING

    def test_step_updates(self):
        """Test step-by-step updates"""
        updates = []
        tracker = ProgressTracker(total_steps=10, callback=lambda u: updates.append(u))

        tracker.update_phase(ProgressPhase.GENERATING, "Starting...")

        # Update steps
        for step in range(1, 11):
            tracker.update_step(step)

        assert len(updates) == 11  # 1 phase update + 10 steps
        assert tracker.current_step == 10

        # Check last update
        last_update = updates[-1]
        assert last_update.step == 10
        assert last_update.total_steps == 10

    def test_progress_calculation(self):
        """Test progress calculation across phases"""
        tracker = ProgressTracker(total_steps=20)

        # Initializing (0-5%)
        tracker.update_phase(ProgressPhase.INITIALIZING, "Init")
        progress = tracker._calculate_total_progress()
        assert 0 <= progress < 5

        # Loading model (5-15%)
        tracker.update_phase(ProgressPhase.LOADING_MODEL, "Loading")
        progress = tracker._calculate_total_progress()
        assert 5 <= progress < 15

        # Generating (20-90%)
        tracker.update_phase(ProgressPhase.GENERATING, "Generating")
        tracker.update_step(10)  # 50% through generation
        progress = tracker._calculate_total_progress()
        assert 40 <= progress <= 60  # Should be around 50%

    def test_complete(self):
        """Test completion"""
        updates = []
        tracker = ProgressTracker(total_steps=20, callback=lambda u: updates.append(u))

        tracker.complete("Done!")

        assert tracker.phase == ProgressPhase.COMPLETE
        assert len(updates) == 1
        assert updates[0].progress == 100.0
        assert updates[0].message == "Done!"

    def test_error(self):
        """Test error handling"""
        updates = []
        tracker = ProgressTracker(total_steps=20, callback=lambda u: updates.append(u))

        tracker.update_phase(ProgressPhase.GENERATING, "Generating...")
        tracker.error("Something went wrong")

        assert tracker.phase == ProgressPhase.ERROR
        assert updates[-1].phase == ProgressPhase.ERROR
        assert "wrong" in updates[-1].message

    def test_eta_estimation(self):
        """Test ETA estimation"""
        tracker = ProgressTracker(total_steps=20)
        tracker.start_time = time.time() - 10  # 10 seconds ago

        tracker.update_phase(ProgressPhase.GENERATING, "Generating")
        tracker.update_step(10)  # 50% done in 10 seconds

        eta = tracker._estimate_eta()

        # Should estimate ~10 seconds remaining
        assert eta is not None
        assert 5 <= eta <= 15  # Rough estimate


class TestStreamingQueue:
    """Test StreamingQueue functionality"""

    @pytest.mark.asyncio
    async def test_put_and_get(self):
        """Test adding and retrieving updates"""
        queue = StreamingQueue()

        update = ProgressUpdate(
            phase=ProgressPhase.GENERATING,
            progress=50.0,
            message="Test"
        )

        queue.put(update)

        retrieved = await queue.get()

        assert retrieved is not None
        assert retrieved.phase == ProgressPhase.GENERATING
        assert retrieved.progress == 50.0

    @pytest.mark.asyncio
    async def test_streaming(self):
        """Test async streaming"""
        queue = StreamingQueue()

        # Add some updates
        for i in range(5):
            queue.put(ProgressUpdate(
                phase=ProgressPhase.GENERATING,
                progress=i * 20.0,
                message=f"Step {i}"
            ))

        queue.mark_done()

        # Collect streamed updates
        updates = []
        async for update in queue.stream():
            updates.append(update)

        assert len(updates) == 5
        assert updates[0].progress == 0.0
        assert updates[-1].progress == 80.0

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Test timeout on empty queue"""
        queue = StreamingQueue()

        # Should return None after timeout
        result = await queue.get()
        assert result is None


class TestSSEHelpers:
    """Test SSE helper functions"""

    def test_create_sse_message(self):
        """Test SSE message creation"""
        data = {"test": "value", "number": 123}
        sse = create_sse_message(data, event="test")

        assert sse.startswith("event: test\n")
        assert "data:" in sse
        assert sse.endswith("\n\n")

        # Verify JSON
        data_line = [line for line in sse.split("\n") if line.startswith("data:")][0]
        json_str = data_line[5:].strip()
        parsed = json.loads(json_str)

        assert parsed == data

    def test_create_sse_heartbeat(self):
        """Test heartbeat creation"""
        heartbeat = create_sse_heartbeat()

        assert heartbeat.startswith(":")
        assert heartbeat.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_sse_generator(self):
        """Test SSE generator"""
        queue = StreamingQueue()

        # Add updates
        queue.put(ProgressUpdate(
            phase=ProgressPhase.GENERATING,
            progress=50.0,
            message="Test 1"
        ))
        queue.put(ProgressUpdate(
            phase=ProgressPhase.COMPLETE,
            progress=100.0,
            message="Done"
        ))
        queue.mark_done()

        # Collect SSE messages
        messages = []
        async for message in sse_generator(queue, heartbeat_interval=999):
            messages.append(message)

        assert len(messages) >= 2  # At least 2 updates + done message

        # Check format
        for message in messages[:-1]:  # Except last (done)
            assert "event:" in message
            assert "data:" in message


class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_full_generation_simulation(self):
        """Simulate full generation with progress tracking"""
        queue = StreamingQueue()
        tracker = ProgressTracker(total_steps=10, callback=queue.put)

        # Simulate generation phases
        tracker.update_phase(ProgressPhase.INITIALIZING, "Initializing...")
        await asyncio.sleep(0.01)

        tracker.update_phase(ProgressPhase.LOADING_MODEL, "Loading model...")
        await asyncio.sleep(0.01)

        tracker.update_phase(ProgressPhase.PREPARING, "Preparing...")
        await asyncio.sleep(0.01)

        tracker.update_phase(ProgressPhase.GENERATING, "Generating...")
        for step in range(1, 11):
            tracker.update_step(step)
            await asyncio.sleep(0.01)

        tracker.update_phase(ProgressPhase.POST_PROCESSING, "Post-processing...")
        await asyncio.sleep(0.01)

        tracker.complete("Generation complete!")
        queue.mark_done()

        # Collect all updates
        updates = []
        async for update in queue.stream():
            updates.append(update)

        # Verify progression
        assert len(updates) > 10
        assert updates[0].phase == ProgressPhase.INITIALIZING
        assert updates[-1].phase == ProgressPhase.COMPLETE
        assert updates[-1].progress == 100.0

        # Verify progress is monotonically increasing (or equal)
        for i in range(1, len(updates)):
            assert updates[i].progress >= updates[i-1].progress


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
