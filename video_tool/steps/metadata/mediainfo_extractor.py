from pathlib import Path
import structlog
from typing import Dict, Any

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata, StepConfig

logger = structlog.get_logger(__name__)

class MediaInfoExtractorStep(BaseStep):
    """
    Extracts comprehensive metadata using MediaInfo.
    This can be an alternative or supplement to FFmpegExtractorStep.
    """
    
    name = "mediainfo_extractor"
    version = "1.0"
    description = "Extract comprehensive video and audio metadata using MediaInfo."
    category = "metadata"
    
    requires = ["file_path"]
    # 'provides' reflects the top-level keys in StepResult.data that map to clips table columns
    # or well-known JSONB structures.
    provides = [
        "width", "height", "duration_seconds", "codec", "container",
        "frame_rate", "bit_rate_kbps", "file_size_bytes",
        "technical_metadata", # For additional details
        "audio_tracks"      # For the JSONB audio_tracks column
    ]

    def __init__(self, config: StepConfig):
        super().__init__(config)
        self.pymediainfo_available = self._check_pymediainfo()

    def _check_pymediainfo(self) -> bool:
        try:
            import pymediainfo
            return True
        except ImportError:
            logger.warning("pymediainfo library not available for MediaInfoExtractorStep.")
            logger.info("Install: pip install pymediainfo")
            return False

    async def process(self, video: VideoMetadata, context: dict = None) -> StepResult:
        if not self.pymediainfo_available:
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video.video_id,
                error="pymediainfo library not installed."
            )

        file_path_str = video.file_path
        file_path = Path(file_path_str)

        if not file_path.exists():
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video.video_id,
                error=f"File not found: {file_path_str}"
            )

        try:
            import pymediainfo # Import here again
            
            media_info = pymediainfo.MediaInfo.parse(file_path_str)
            
            general_track_info = None
            video_track_info = None
            parsed_audio_tracks = []

            for track in media_info.tracks:
                if track.track_type == 'General' and not general_track_info:
                    general_track_info = track.to_data()
                elif track.track_type == 'Video' and not video_track_info:
                    video_track_info = track.to_data()
                elif track.track_type == 'Audio':
                    parsed_audio_tracks.append(track.to_data())
            
            output_data = {}
            technical_details = {} # For less common fields to nest under technical_metadata

            if general_track_info:
                if general_track_info.get('duration'):
                    try:
                        output_data['duration_seconds'] = float(general_track_info['duration']) / 1000.0
                    except ValueError:
                        logger.warning("Could not parse general track duration", value=general_track_info['duration'])
                if general_track_info.get('overall_bit_rate'):
                    try:
                        # MediaInfo often gives bit rate in bps, convert to kbps if needed or store as is if already kbps
                        # Assuming it's bps for now, needs verification based on pymediainfo output
                        output_data['bit_rate_kbps'] = int(general_track_info['overall_bit_rate']) / 1000
                    except ValueError:
                         logger.warning("Could not parse overall_bit_rate", value=general_track_info['overall_bit_rate'])
                if general_track_info.get('file_size'):
                    output_data['file_size_bytes'] = int(general_track_info['file_size'])
                if general_track_info.get('format'):
                    output_data['container'] = general_track_info['format']
                
                # Store other general info in technical_details
                technical_details['mediainfo_general'] = {
                    k: v for k, v in general_track_info.items()
                    if k not in ['duration', 'overall_bit_rate', 'file_size', 'format']
                }


            if video_track_info:
                output_data['width'] = video_track_info.get('width')
                output_data['height'] = video_track_info.get('height')
                output_data['codec'] = video_track_info.get('codec_id') or video_track_info.get('format')
                if video_track_info.get('frame_rate'):
                    try:
                        output_data['frame_rate'] = float(video_track_info['frame_rate'])
                    except ValueError:
                        logger.warning("Could not parse frame_rate", value=video_track_info['frame_rate'])

                if video_track_info.get('bit_depth'): # Often string like "8 bits"
                    try:
                        output_data['bit_depth'] = int(str(video_track_info['bit_depth']).split()[0])
                    except (ValueError, IndexError):
                         logger.warning("Could not parse bit_depth", value=video_track_info['bit_depth'])
                
                output_data['color_space'] = video_track_info.get('color_space')
                
                # If video track has a more specific bit_rate, prefer it
                if video_track_info.get('bit_rate'):
                    try:
                        output_data['bit_rate_kbps'] = int(video_track_info['bit_rate']) / 1000 # Assuming bps
                    except ValueError:
                        logger.warning("Could not parse video track bit_rate", value=video_track_info['bit_rate'])

                # Store other video info in technical_details
                technical_details['mediainfo_video'] = {
                    k: v for k, v in video_track_info.items()
                    if k not in ['width', 'height', 'codec_id', 'format', 'frame_rate', 'bit_depth', 'color_space', 'bit_rate']
                }

            if technical_details:
                output_data['technical_metadata'] = technical_details
            
            if parsed_audio_tracks:
                output_data['audio_tracks'] = parsed_audio_tracks

            # Filter out None values from the top-level output_data
            final_output_data = {k: v for k, v in output_data.items() if v is not None}

            logger.info(f"Extracted MediaInfo metadata for {video.file_name}", data_keys=list(final_output_data.keys()))
            
            return StepResult(
                success=True,
                step_name=self.name,
                video_id=video.video_id,
                data=final_output_data
            )
            
        except Exception as e:
            logger.error(f"MediaInfoExtractorStep failed for {video.file_name}: {str(e)}", exc_info=True)
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video.video_id,
                error=str(e)
            )

    async def setup(self):
        if self.pymediainfo_available:
            logger.info("MediaInfoExtractorStep initialized and pymediainfo library is available.")
        else:
            logger.warning("MediaInfoExtractorStep initialized, but pymediainfo library is NOT available.")