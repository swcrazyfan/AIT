from typing import Dict, Any, List, Optional
from pathlib import Path
import structlog
import asyncio
from typing import List, Dict, Any

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata, StepConfig

logger = structlog.get_logger(__name__)

class FFmpegThumbsStep(BaseStep):
    """
    Generates thumbnails using FFmpeg.
    Often faster than OpenCV for simple thumbnail generation.
    """
    
    name = "ffmpeg_thumbs" # As per fast.yaml
    version = "1.0"
    description = "Generate thumbnails efficiently using FFmpeg."
    category = "thumbnails"
    
    requires = ["file_path", "duration_seconds"]
    provides = ["thumbnails", "technical_metadata"] # Updated provides

    def __init__(self, config: StepConfig):
        super().__init__(config)
        # FFmpeg is assumed to be in PATH, no specific library check here,
        # but a real implementation might check for ffmpeg executable.

    async def process(self, video: VideoMetadata, context: dict = None) -> StepResult:
        file_path_str = video.file_path
        video_id = video.video_id
        
        count = self.config.params.get("count", 3)
        width = self.config.params.get("width", 1280) # From fast.yaml
        
        output_base_dir = Path("thumbnails_output") # Example, should be configured
        output_dir = output_base_dir / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        thumbnail_paths: List[str] = []
        tasks = []

        duration = video.duration_seconds
        if not duration or duration <= 0:
            return StepResult(success=False, step_name=self.name, video_id=video_id, error="Video duration not available or invalid.")

        try:
            for i in range(count):
                # Calculate timestamp for each thumbnail
                # Example: equally spaced, avoiding start/end too much
                timestamp = duration * (i + 1) / (count + 2) 
                
                thumb_filename = f"ffmpeg_thumb_{i:03d}_{int(timestamp)}s.jpg"
                thumb_path = output_dir / thumb_filename
                
                # FFmpeg command:
                # -ss: seek to timestamp
                # -i: input file
                # -frames:v 1: extract one video frame
                # -vf scale={width}:-1 : scale to specified width, maintain aspect ratio
                # -q:v 2 : output quality (2-5 is good for JPEG)
                cmd = [
                    "ffmpeg",
                    "-ss", str(timestamp),
                    "-i", file_path_str,
                    "-frames:v", "1",
                    "-vf", f"scale={width}:-1",
                    "-q:v", "2", # JPEG quality, 2 is high
                    str(thumb_path)
                ]
                tasks.append(self._run_ffmpeg_command(cmd, str(thumb_path)))

            # Run all FFmpeg commands in parallel
            generated_paths = await asyncio.gather(*tasks)
            thumbnail_paths = [p for p in generated_paths if p is not None]

            if not thumbnail_paths and count > 0:
                 logger.warning(f"FFmpegThumbsStep: No thumbnails generated for {video.file_name}, though {count} were requested.")


            logger.info(f"Generated {len(thumbnail_paths)} FFmpeg thumbnails for {video.file_name}")
            
            return StepResult(
                success=True,
                step_name=self.name,
                video_id=video_id,
                data={
                    "thumbnails": thumbnail_paths, # Direct column in 'clips'
                    "technical_metadata": {
                        "thumbnail_details": { # Nested details
                            "thumbnail_count": len(thumbnail_paths),
                            "generation_method": "ffmpeg"
                        }
                    }
                },
                artifacts={f"ffmpeg_thumbnail_{j}": p for j, p in enumerate(thumbnail_paths)}
            )
            
        except Exception as e:
            logger.error(f"FFmpegThumbsStep failed for {video.file_name}: {str(e)}", exc_info=True)
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video_id,
                error=str(e)
            )

    async def _run_ffmpeg_command(self, cmd: List[str], output_path: str) -> Optional[str]:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logger.debug(f"FFmpeg command successful for {output_path}")
            return output_path
        else:
            logger.error(f"FFmpeg command failed for {output_path}: {stderr.decode()}")
            return None

    async def setup(self):
        # Could add a check here if ffmpeg executable is found in PATH
        logger.info("FFmpegThumbsStep initialized. Assumes FFmpeg is in PATH.")