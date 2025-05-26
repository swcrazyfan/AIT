from pathlib import Path
import structlog
import asyncio
from typing import Dict, Any, Optional

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata, StepConfig

logger = structlog.get_logger(__name__)

class FFmpegCompressStep(BaseStep):
    """
    Compresses a video file using FFmpeg.
    """
    
    name = "ffmpeg_compress" # As per default.yaml and ai_research.yaml
    version = "1.0"
    description = "Compress video using FFmpeg with configurable parameters."
    category = "compression"
    
    requires = ["file_path"] # Needs the original file path
    provides = ["technical_metadata"] # Compression details will be nested here

    def __init__(self, config: StepConfig):
        super().__init__(config)
        # FFmpeg is assumed to be in PATH

    async def process(self, video: VideoMetadata, context: dict = None) -> StepResult:
        original_file_path_str = video.file_path
        video_id = video.video_id
        
        # Get compression parameters from config, with defaults
        codec = self.config.params.get("codec", "libx264") # Default to widely compatible H.264
        bitrate = self.config.params.get("bitrate", "1000k") # Default bitrate
        fps = self.config.params.get("fps") # Optional: change fps
        crf = self.config.params.get("crf", 23) # Constant Rate Factor (for libx264, libx265)
        preset = self.config.params.get("preset", "medium") # FFmpeg preset
        output_suffix = self.config.params.get("suffix", "_compressed")
        
        original_path = Path(original_file_path_str)
        
        # Define output path (e.g., in a run-specific or global cache)
        # A more robust solution would use the OutputManager.
        output_base_dir = Path("compressed_videos") # Example, should be configured
        output_dir = output_base_dir / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        compressed_filename = f"{original_path.stem}{output_suffix}{original_path.suffix}"
        compressed_file_path = output_dir / compressed_filename

        try:
            cmd = [
                "ffmpeg",
                "-i", original_file_path_str,
                "-c:v", codec,
                "-b:v", bitrate,
                "-preset", preset,
            ]

            if codec in ["libx264", "libx265"]: # CRF is specific to these codecs
                cmd.extend(["-crf", str(crf)])
            
            if fps:
                cmd.extend(["-r", str(fps)])

            # Add other parameters as needed, e.g., -c:a for audio codec
            cmd.extend(["-y", str(compressed_file_path)]) # -y to overwrite output file if it exists

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"FFmpeg compression failed for {original_path.name}: {stderr.decode()}")
                return StepResult(
                    success=False,
                    step_name=self.name,
                    video_id=video_id,
                    error=f"FFmpeg compression failed: {stderr.decode()}"
                )

            compressed_size_bytes = compressed_file_path.stat().st_size
            logger.info(f"Successfully compressed {original_path.name} to {compressed_file_path} ({compressed_size_bytes} bytes)")
            
            return StepResult(
                success=True,
                step_name=self.name,
                video_id=video_id,
                data={
                    "technical_metadata": {
                        "compression_details": {
                            "compressed_file_size_bytes": compressed_size_bytes,
                            "compression_codec": codec,
                            "compression_bitrate": bitrate,
                            "compression_fps": fps,
                            "compression_crf": crf if codec in ["libx264", "libx265"] else None,
                            "compression_preset": preset
                        }
                    }
                },
                artifacts={"compressed_video": str(compressed_file_path)} # This will be saved to 'artifacts' table
            )
            
        except Exception as e:
            logger.error(f"FFmpegCompressStep failed for {original_path.name}: {str(e)}", exc_info=True)
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video_id,
                error=str(e)
            )

    async def setup(self):
        logger.info("FFmpegCompressStep initialized. Assumes FFmpeg is in PATH.")