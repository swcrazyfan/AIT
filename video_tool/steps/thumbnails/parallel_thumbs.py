from typing import Dict, Any, List, Optional
import asyncio
from pathlib import Path
from typing import List, Optional
import cv2

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata

class ParallelThumbsStep(BaseStep):
    """Generate thumbnails in parallel with immediate saves"""
    
    name = "parallel_thumbs"
    version = "1.0"
    description = "Generate thumbnails in parallel"
    category = "thumbnails"
    
    requires = ["file_path", "duration_seconds", "user_id"] # user_id is needed for _generate_single_thumbnail
    provides = ["thumbnails", "technical_metadata"] # Updated provides
    save_partial = True # Indicates that _generate_single_thumbnail performs partial saves
    
    async def process(self, video: VideoMetadata) -> StepResult:
        try:
            count = self.config.params.get("count", 5)
            width = self.config.params.get("width", 1920)
            
            if video.duration_seconds is None:
                self.logger.error(f"Video duration is not available for video_id: {video.video_id}")
                return StepResult(
                    success=False,
                    step_name=self.name,
                    video_id=video.video_id,
                    error="Video duration is not available."
                )
            
            # Calculate thumbnail positions in seconds
            duration_sec = video.duration_seconds
            # Ensure count is at least 1 to avoid division by zero if count is 0 or negative
            effective_count = max(1, count)
            positions_sec = [duration_sec * (i + 1) / (effective_count + 1) for i in range(effective_count)]

            if not video.user_id:
                self.logger.error(f"User ID is not available in VideoMetadata for video_id: {video.video_id}")
                return StepResult(
                    success=False,
                    step_name=self.name,
                    video_id=video.video_id,
                    error="User ID not found in video metadata."
                )

            # Generate thumbnails in parallel
            tasks = []
            for i, pos_sec in enumerate(positions_sec):
                task = self._generate_single_thumbnail(
                    video_id=video.video_id,
                    user_id=video.user_id,
                    video_path=video.file_path,
                    position_ms=pos_sec * 1000, # Convert seconds to milliseconds
                    index=i,
                    width=width
                )
                tasks.append(task)
            
            # Run all thumbnail generations in parallel
            thumbnail_paths = await asyncio.gather(*tasks)
            
            # Filter out any failed thumbnails
            valid_thumbnails = [t for t in thumbnail_paths if t is not None]
            
            # The main 'thumbnails' field in the clips table will be updated by BaseStep._save_step_results
            # if save_immediately is True (which it is by default).
            # The RPC call in _generate_single_thumbnail handles the incremental append.
            return StepResult(
                success=True,
                step_name=self.name,
                video_id=video.video_id,
                data={
                    "thumbnails": valid_thumbnails, # Direct column in 'clips'
                    "technical_metadata": {
                        "thumbnail_details": { # Nested details
                            "thumbnail_count": len(valid_thumbnails),
                            "generation_method": "opencv_parallel" # Distinguish from single OpenCV
                        }
                    }
                },
                artifacts={
                    f"thumbnail_{i}": path
                    for i, path in enumerate(valid_thumbnails)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in ParallelThumbsStep process for video {video.video_id}: {str(e)}", exc_info=True)
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video.video_id,
                error=str(e)
            )
    
    async def _generate_single_thumbnail(self, video_id: str, user_id: str, video_path: str, position_ms: float, index: int, width: int) -> Optional[str]:
        """Generate a single thumbnail and save its path immediately via RPC"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Failed to open video: {video_path} for thumbnail {index} of video {video_id}")
                return None
            
            cap.set(cv2.CAP_PROP_POS_MSEC, position_ms)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                self.logger.error(f"Failed to read frame at position {position_ms:.2f}ms for video {video_id}, thumbnail {index}")
                return None
            
            original_height, original_width = frame.shape[:2]
            if original_width == 0:
                 self.logger.error(f"Invalid frame width (0) for video {video_id} at position {position_ms:.2f}ms, thumbnail {index}")
                 return None
            
            aspect_ratio = original_height / original_width
            height = int(width * aspect_ratio)
            resized_frame = cv2.resize(frame, (width, height))
            
            # Define output path. This should ideally be managed by an OutputManager or similar.
            # For now, using a relative path as per the plan's example.
            output_dir = Path(f"thumbnails/{video_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = f"thumb_{index:03d}_{int(position_ms/1000)}s.jpg"
            output_path = output_dir / output_filename
            
            cv2.imwrite(str(output_path), resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Update database immediately with this thumbnail path
            if self.supabase: # Initialized in BaseStep.execute()
                # The RPC function 'append_to_array' is defined in the SQL schema.
                # It appends a value to a JSONB array column in a specified table.
                # RLS on 'clips' table handles user-specific context.
                rpc_params = {
                    'table_name': 'clips',
                    'id': video_id, 
                    'column_name': 'thumbnails', # The array column to append to
                    'new_value': str(output_path) 
                }
                response = await self.supabase.rpc('append_to_array', rpc_params).execute()
                if response.error:
                    self.logger.error(f"RPC append_to_array failed for video {video_id}, thumbnail {output_path}: {response.error.message}")
                    # Decide if this is a critical failure for the thumbnail generation itself.
                    # For now, the local file is created, so we might still return its path.
            else:
                self.logger.error(f"Supabase client not initialized. Cannot save partial thumbnail result for video {video_id}, thumbnail {index}.")
            
            self.logger.info(f"Generated thumbnail {index} at {position_ms/1000:.2f}s for video {video_id}: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate thumbnail {index} for video {video_id} at {position_ms:.2f}ms: {str(e)}", exc_info=True)
            return None