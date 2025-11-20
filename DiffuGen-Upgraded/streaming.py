"""
Streaming Support Module for DiffuGen
Provides Server-Sent Events (SSE) for real-time progress updates
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Optional, Callable, Dict, Any, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ProgressPhase(Enum):
    """Generation progress phases"""
    INITIALIZING = "initializing"
    LOADING_MODEL = "loading_model"
    PREPARING = "preparing"
    GENERATING = "generating"
    POST_PROCESSING = "post_processing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ProgressUpdate:
    """Progress update data"""
    phase: ProgressPhase
    progress: float  # 0-100
    message: str
    step: Optional[int] = None
    total_steps: Optional[int] = None
    eta_seconds: Optional[float] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            "phase": self.phase.value,
            "progress": self.progress,
            "message": self.message,
            "timestamp": self.timestamp
        }

        if self.step is not None:
            data["step"] = self.step
        if self.total_steps is not None:
            data["total_steps"] = self.total_steps
        if self.eta_seconds is not None:
            data["eta_seconds"] = self.eta_seconds

        return data

    def to_sse(self, event_type: str = "progress") -> str:
        """Convert to Server-Sent Events format"""
        data_json = json.dumps(self.to_dict())
        return f"event: {event_type}\ndata: {data_json}\n\n"


class ProgressTracker:
    """
    Tracks generation progress and sends updates

    Since sd.cpp doesn't provide real-time progress, we simulate
    progress based on known phases and timing estimates
    """

    def __init__(
        self,
        total_steps: int = 20,
        callback: Optional[Callable[[ProgressUpdate], None]] = None
    ):
        self.total_steps = total_steps
        self.callback = callback
        self.current_step = 0
        self.start_time = time.time()
        self.phase = ProgressPhase.INITIALIZING

        # Progress allocation per phase (total = 100%)
        self.phase_weights = {
            ProgressPhase.INITIALIZING: 5,
            ProgressPhase.LOADING_MODEL: 10,
            ProgressPhase.PREPARING: 5,
            ProgressPhase.GENERATING: 70,  # Most time spent here
            ProgressPhase.POST_PROCESSING: 10
        }

        self.phase_progress = 0  # Current progress within phase

    def _calculate_total_progress(self) -> float:
        """Calculate total progress across all phases"""
        # Sum of completed phase weights
        completed_weight = sum(
            weight for phase, weight in self.phase_weights.items()
            if phase.value < self.phase.value
        )

        # Add current phase progress
        current_phase_weight = self.phase_weights.get(self.phase, 0)
        current_contribution = (self.phase_progress / 100.0) * current_phase_weight

        return completed_weight + current_contribution

    def _estimate_eta(self) -> Optional[float]:
        """Estimate time remaining"""
        elapsed = time.time() - self.start_time
        progress = self._calculate_total_progress()

        if progress < 5:  # Not enough data yet
            return None

        # Simple linear estimation
        total_estimated = (elapsed / progress) * 100
        remaining = total_estimated - elapsed

        return max(0, remaining)

    def _send_update(self, update: ProgressUpdate):
        """Send progress update to callback"""
        if self.callback:
            try:
                self.callback(update)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    def update_phase(self, phase: ProgressPhase, message: str):
        """Update current phase"""
        self.phase = phase
        self.phase_progress = 0

        update = ProgressUpdate(
            phase=phase,
            progress=self._calculate_total_progress(),
            message=message,
            eta_seconds=self._estimate_eta()
        )

        logger.info(f"Progress: {phase.value} - {message}")
        self._send_update(update)

    def update_step(self, step: int, message: Optional[str] = None):
        """Update step within generation phase"""
        self.current_step = step
        self.phase_progress = (step / self.total_steps) * 100

        if message is None:
            message = f"Generating step {step}/{self.total_steps}"

        update = ProgressUpdate(
            phase=self.phase,
            progress=self._calculate_total_progress(),
            message=message,
            step=step,
            total_steps=self.total_steps,
            eta_seconds=self._estimate_eta()
        )

        self._send_update(update)

    def update_progress(self, progress_percent: float, message: str):
        """Update progress within current phase"""
        self.phase_progress = progress_percent

        update = ProgressUpdate(
            phase=self.phase,
            progress=self._calculate_total_progress(),
            message=message,
            eta_seconds=self._estimate_eta()
        )

        self._send_update(update)

    def complete(self, message: str = "Generation complete"):
        """Mark as complete"""
        self.phase = ProgressPhase.COMPLETE

        update = ProgressUpdate(
            phase=ProgressPhase.COMPLETE,
            progress=100.0,
            message=message
        )

        logger.info(f"Progress: Complete - {message}")
        self._send_update(update)

    def error(self, message: str):
        """Mark as error"""
        self.phase = ProgressPhase.ERROR

        update = ProgressUpdate(
            phase=ProgressPhase.ERROR,
            progress=self._calculate_total_progress(),
            message=message
        )

        logger.error(f"Progress: Error - {message}")
        self._send_update(update)


class StreamingQueue:
    """
    Async queue for streaming progress updates

    Used to bridge sync progress callbacks to async SSE streams
    """

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.done = False

    def put(self, update: ProgressUpdate):
        """Add update to queue (sync)"""
        try:
            # Get or create event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule from sync context
                asyncio.run_coroutine_threadsafe(
                    self.queue.put(update),
                    loop
                )
            else:
                # Direct put if loop not running
                asyncio.create_task(self.queue.put(update))
        except Exception as e:
            logger.error(f"Error adding to streaming queue: {e}")

    async def get(self) -> Optional[ProgressUpdate]:
        """Get update from queue (async)"""
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    def mark_done(self):
        """Mark stream as done"""
        self.done = True

    async def stream(self) -> AsyncGenerator[ProgressUpdate, None]:
        """Stream updates until done"""
        while not self.done or not self.queue.empty():
            update = await self.get()
            if update:
                yield update


# ============================================================================
# SSE Helpers
# ============================================================================

def create_sse_message(data: Dict[str, Any], event: str = "message") -> str:
    """Create Server-Sent Events formatted message"""
    data_json = json.dumps(data)
    return f"event: {event}\ndata: {data_json}\n\n"


def create_sse_heartbeat() -> str:
    """Create SSE heartbeat (comment)"""
    return ": heartbeat\n\n"


async def sse_generator(
    streaming_queue: StreamingQueue,
    heartbeat_interval: float = 15.0
) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream from streaming queue

    Args:
        streaming_queue: Queue of progress updates
        heartbeat_interval: Seconds between heartbeat messages

    Yields:
        SSE formatted strings
    """
    last_heartbeat = time.time()

    try:
        async for update in streaming_queue.stream():
            # Send progress update
            yield update.to_sse()

            # Send heartbeat if needed
            now = time.time()
            if now - last_heartbeat > heartbeat_interval:
                yield create_sse_heartbeat()
                last_heartbeat = now

        # Send final done message
        yield create_sse_message({"done": True}, event="done")

    except Exception as e:
        logger.error(f"Error in SSE generator: {e}")
        # Send error message
        error_msg = create_sse_message({
            "error": str(e),
            "phase": "error"
        }, event="error")
        yield error_msg


# ============================================================================
# Testing Utilities
# ============================================================================

async def test_progress_tracker():
    """Test progress tracker with simulated generation"""
    print("Testing ProgressTracker...\n")

    updates = []

    def collect_update(update: ProgressUpdate):
        updates.append(update)
        print(f"[{update.phase.value:20s}] {update.progress:5.1f}% - {update.message}")

    tracker = ProgressTracker(total_steps=20, callback=collect_update)

    # Simulate generation phases
    tracker.update_phase(ProgressPhase.INITIALIZING, "Initializing...")
    await asyncio.sleep(0.5)

    tracker.update_phase(ProgressPhase.LOADING_MODEL, "Loading model...")
    await asyncio.sleep(1.0)

    tracker.update_phase(ProgressPhase.PREPARING, "Preparing generation...")
    await asyncio.sleep(0.5)

    tracker.update_phase(ProgressPhase.GENERATING, "Generating image...")
    for step in range(1, 21):
        tracker.update_step(step)
        await asyncio.sleep(0.2)

    tracker.update_phase(ProgressPhase.POST_PROCESSING, "Post-processing...")
    await asyncio.sleep(0.5)

    tracker.complete()

    print(f"\nTotal updates: {len(updates)}")
    print(f"Final progress: {updates[-1].progress}%")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_progress_tracker())
