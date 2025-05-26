from typing import Dict, Any, List, Optional
from pathlib import Path
import structlog
from typing import List, Optional, Dict, Any
import asyncio # For potential async operations if cv2 allows or for managing multiple frames

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata, StepConfig

logger = structlog.get_logger(__name__)

class OpenCVThumbsStep(BaseStep):
    """
    Generates thumbnails using OpenCV.
    Provides more control over frame extraction and processing than FFmpeg.
    This is an alternative to the existing ParallelThumbsStep which also uses OpenCV.
    This specific file might be for a different strategy or configuration.
    """
    
    name = "opencv_thumbs" # As per architecture diagram
    version = "1.0"
    description = "Generate thumbnails using OpenCV with specific frame selection."
    category = "thumbnails"
    
    requires = ["file_path", "duration_seconds"] # duration_seconds needed for positioning
    provides = ["thumbnails", "technical_metadata"] # Updated provides

    def __init__(self, config: StepConfig):
        super().__init__(config)
        self.opencv_available = self._check_opencv()

    def _check_opencv(self) -> bool:
        try:
            import cv2
            return True
        except ImportError:
            logger.warning("OpenCV (cv2) library not available for OpenCVThumbsStep.")
            logger.info("Install: pip install opencv-python")
            return False

    async def process(self, video: VideoMetadata, context: dict = None) -> StepResult:
        if not self.opencv_available:
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video.video_id,
                error="OpenCV (cv2) library not installed."
            )

        file_path_str = video.file_path
        video_id = video.video_id
        
        # Default parameters, can be overridden by step config
        count = self.config.params.get("count", 3)
        width = self.config.params.get("width", 640) # Smaller default for a potentially different use case
        output_quality = self.config.params.get("quality", 85)
        
        # Ensure output directory exists (e.g., in a run-specific or global cache)
        # For this placeholder, let's assume a simple structure.
        # A more robust solution would use the OutputManager.
        output_base_dir = Path("thumbnails_output") # Example, should be configured
        output_dir = output_base_dir / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        thumbnail_paths: List[str] = []
        
        try:
            import cv2 # Import here again

            cap = cv2.VideoCapture(file_path_str)
            if not cap.isOpened():
                return StepResult(success=False, step_name=self.name, video_id=video_id, error="Could not open video file with OpenCV.")

            duration = video.duration_seconds
            if not duration or duration <= 0:
                cap.release()
                return StepResult(success=False, step_name=self.name, video_id=video_id, error="Video duration not available or invalid.")

            positions_ms = [duration * 1000 * i / (count + 1) for i in range(1, count + 1)]

            for i, pos_ms in enumerate(positions_ms):
                cap.set(cv2.CAP_PROP_POS_MSEC, pos_ms)
                ret, frame = cap.read()
                if ret:
                    # Resize frame
                    original_height, original_width = frame.shape[:2]
                    aspect_ratio = original_width / original_height
                    height = int(width / aspect_ratio)
                    resized_frame = cv2.resize(frame, (width, height))
                    
                    # Save thumbnail
                    thumb_filename = f"opencv_thumb_{i:03d}_{int(pos_ms/1000)}s.jpg"
                    thumb_path = output_dir / thumb_filename
                    cv2.imwrite(str(thumb_path), resized_frame, [cv2.IMWRITE_JPEG_QUALITY, output_quality])
                    thumbnail_paths.append(str(thumb_path))
                else:
                    logger.warning(f"Could not read frame at {pos_ms}ms for {video.file_name}")
            
            cap.release()

            logger.info(f"Generated {len(thumbnail_paths)} OpenCV thumbnails for {video.file_name}")
            
            return StepResult(
                success=True,
                step_name=self.name,
                video_id=video_id,
                data={
                    "thumbnails": thumbnail_paths, # Direct column in 'clips'
                    "technical_metadata": {
                        "thumbnail_details": { # Nested details
                            "thumbnail_count": len(thumbnail_paths),
                            "generation_method": "opencv"
                        }
                    }
                },
                artifacts={f"opencv_thumbnail_{j}": p for j, p in enumerate(thumbnail_paths)}
            )
            
        except Exception as e:
            logger.error(f"OpenCVThumbsStep failed for {video.file_name}: {str(e)}", exc_info=True)
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video_id,
                error=str(e)
            )

    async def setup(self):
        if self.opencv_available:
            logger.info("OpenCVThumbsStep initialized and OpenCV (cv2) library is available.")
        else:
            logger.warning("OpenCVThumbsStep initialized, but OpenCV (cv2) library is NOT available.")