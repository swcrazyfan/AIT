import asyncio
import json
from pathlib import Path
from typing import Dict

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata

class FFmpegExtractorStep(BaseStep):
    """Extract metadata using ffprobe with incremental saves"""

    name = "ffmpeg_extractor"
    version = "1.0"
    description = "Extract video metadata using ffprobe"
    category = "metadata"

    requires = ["file_path"]
    provides = ["duration_seconds", "width", "height", "fps", "codec", "bit_rate_kbps", "container", "technical_metadata"] # Updated provides
    save_partial = True  # Enable partial saves
 
    async def process(self, video: VideoMetadata, context: dict = None) -> StepResult:
        try:
            file_path = video.file_path
            self.logger.info(f"FFmpegExtractorStep processing file: {file_path}", video_id=video.video_id) # Log input path

            # First, get basic info quickly
            basic_info = await self._get_basic_info(file_path)

            # Save basic info immediately if save_partial is True
            if self.save_partial:
                await self.save_partial_result(video.video_id, video.user_id, basic_info)

            # Then get detailed info
            detailed_info = await self._get_detailed_info(file_path)

            # Combine all metadata
            metadata = {**basic_info, **detailed_info}
            self.logger.info(f"FFmpegExtractorStep extracted metadata for {file_path}: {metadata}", video_id=video.video_id) # Log extracted metadata

            return StepResult(
                success=True,
                step_name=self.name,
                video_id=video.video_id,
                data=metadata
            )

        except Exception as e:
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video.video_id,
                error=str(e)
            )

    async def _get_basic_info(self, file_path: str) -> Dict:
        """Get basic info quickly"""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            file_path
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, _ = await proc.communicate()
        probe_data = json.loads(stdout.decode())
        format_data = probe_data.get("format", {})

        bitrate_bps = int(format_data.get("bit_rate", 0))
        return {
            "duration_seconds": float(format_data.get("duration", 0)),
            "bit_rate_kbps": bitrate_bps // 1000 if bitrate_bps else 0, # Convert to kbps
            "container": format_data.get("format_name") # Renamed from format_name
        }

    async def _get_detailed_info(self, file_path: str) -> Dict:
        """Get detailed stream info"""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            file_path
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, _ = await proc.communicate()
        probe_data = json.loads(stdout.decode())

        # Extract video stream info
        video_stream = next(
            (s for s in probe_data.get("streams", []) if s["codec_type"] == "video"),
            {}
        )

        # Extract audio info
        audio_streams = [s for s in probe_data.get("streams", []) if s["codec_type"] == "audio"]

        # Direct mappings to 'clips' table columns
        direct_metadata = {
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "codec": video_stream.get("codec_name"),
        }
        
        # Details to be nested under 'technical_metadata'
        ffmpeg_specific_details = {
            "has_audio": len(audio_streams) > 0,
            "audio_tracks_count": len(audio_streams),
            "video_stream_details": video_stream, # Store raw video stream for more details if needed
            "audio_stream_details": audio_streams # Store raw audio streams
        }

        # Calculate FPS
        fps_value = None
        if "r_frame_rate" in video_stream and video_stream["r_frame_rate"] and "/" in video_stream["r_frame_rate"]:
            try:
                num, den = map(int, video_stream["r_frame_rate"].split("/"))
                fps_value = num / den if den != 0 else 0
            except ValueError:
                self.logger.warning(f"Could not parse r_frame_rate: {video_stream['r_frame_rate']}")
                fps_value = None # Or some default / error indicator
        else:
            fps_value = None
        
        if fps_value is not None:
            direct_metadata["fps"] = fps_value

        # Combine direct metadata with nested technical details
        final_metadata = direct_metadata
        if ffmpeg_specific_details:
            final_metadata["technical_metadata"] = {"ffmpeg_details": ffmpeg_specific_details}
            
        return final_metadata