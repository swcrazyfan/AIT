from pathlib import Path
from typing import Dict, Any, Optional
import structlog

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata

class HDRExtractorStep(BaseStep):
    """Extract HDR metadata and color information using PyMediaInfo"""
    
    name = "hdr_extractor"
    version = "1.0"
    description = "Extract HDR metadata, color space, and transfer characteristics"
    category = "metadata"
    
    requires = ["file_path"]
    provides = [
        "technical_metadata", # Main output for structured HDR and color info
        "bit_depth" # Assuming bit_depth (from video track) is a direct column
    ]
    
    def __init__(self, config):
        super().__init__(config) 
        self.pymediainfo_available = self._check_pymediainfo()
        
    def _check_pymediainfo(self) -> bool:
        """Check if PyMediaInfo is available"""
        try:
            import pymediainfo
            self.logger.info("PyMediaInfo library found.")
            return True
        except ImportError:
            self.logger.warning("PyMediaInfo not available for HDR metadata extraction.")
            self.logger.info("To enable HDR extraction, please install PyMediaInfo: pip install pymediainfo")
            return False
    
    async def process(self, video: VideoMetadata, context: Optional[Dict[str, Any]] = None) -> StepResult:
        """Extract HDR metadata from video file"""
        try:
            if not self.pymediainfo_available:
                return StepResult(
                    success=False,
                    step_name=self.name,
                    video_id=video.video_id,
                    error="PyMediaInfo not available for HDR metadata extraction"
                )
            
            file_path = video.file_path
            
            if not Path(file_path).exists():
                return StepResult(
                    success=False,
                    step_name=self.name,
                    video_id=video.video_id,
                    error=f"File not found: {file_path}"
                )
            
            # Extract HDR metadata
            hdr_metadata = await self._extract_hdr_metadata(file_path)
            
            # Also extract codec parameters for enhanced metadata
            codec_metadata = await self._extract_codec_parameters(file_path)
            
            # Combine all metadata
            # Structure the data according to EnhancedVideoIngestOutput and its nested models
            
            output_data = {}
            
            # HDR Details
            hdr_details_data = {
                "is_hdr": hdr_metadata.get("is_hdr", False),
                "hdr_format": hdr_metadata.get("hdr_format"),
                "hdr_format_commercial": hdr_metadata.get("hdr_format_commercial"),
                "transfer_characteristics": hdr_metadata.get("transfer_characteristics"), # Also part of color_details
                "master_display": hdr_metadata.get("master_display"),
                "max_cll": hdr_metadata.get("max_cll"),
                "max_fall": hdr_metadata.get("max_fall")
            }
            # Filter None from hdr_details_data before adding to color_details
            hdr_details_data_filtered = {k: v for k, v in hdr_details_data.items() if v is not None}


            # Color Details (which includes HDR details)
            color_details_data = {
                "color_space": hdr_metadata.get("color_space"),
                "color_primaries": hdr_metadata.get("color_primaries"),
                "transfer_characteristics": hdr_metadata.get("transfer_characteristics"),
                "matrix_coefficients": hdr_metadata.get("matrix_coefficients"),
                "color_range": hdr_metadata.get("color_range"),
                "hdr": hdr_details_data_filtered # Nest the filtered HDR details
            }
            # Filter None from color_details_data
            color_details_data_filtered = {k: v for k, v in color_details_data.items() if v is not None}

            # Technical Metadata (to hold color details and other codec params)
            technical_metadata_payload = {
                "color_details": color_details_data_filtered,
                "codec_params_hdr_step": codec_metadata # Store codec params under a specific key
            }
            
            output_data["technical_metadata"] = {k: v for k, v in technical_metadata_payload.items() if v is not None and v != {}}

            # Direct column for bit_depth if available from codec_metadata
            if codec_metadata.get("bit_depth_video") is not None:
                output_data["bit_depth"] = codec_metadata.get("bit_depth_video")

            # Log important findings
            if hdr_details_data_filtered.get("is_hdr"):
                hdr_format_detected = hdr_details_data_filtered.get("hdr_format", "Unknown")
                self.logger.info(f"HDR video detected: {hdr_format_detected}", video_id=video.video_id)
            else:
                self.logger.info("No HDR video detected or HDR metadata not found.", video_id=video.video_id)
            
            final_output_data = {k: v for k, v in output_data.items() if v is not None}

            return StepResult(
                success=True,
                step_name=self.name,
                video_id=video.video_id,
                data=final_output_data,
                metadata={
                    "extraction_method": "pymediainfo",
                    "hdr_detected": hdr_details_data_filtered.get("is_hdr", False),
                    "bit_depth_from_hdr_step": codec_metadata.get("bit_depth_video")
                }
            )
            
        except Exception as e:
            self.logger.error("Error during HDR metadata extraction", exc_info=True, video_id=video.video_id)
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video.video_id,
                error=str(e)
            )
    
    async def _extract_hdr_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract HDR metadata from video file using PyMediaInfo.
        
        Args:
            file_path: Path to the video file
            
        Returns:
            Dict with HDR metadata
        """
        import pymediainfo # Import locally as it's an optional dependency
        
        try:
            # Parse media file
            media_info = pymediainfo.MediaInfo.parse(file_path)
            
            # Find the first video track
            video_track = next((track for track in media_info.tracks if track.track_type == 'Video'), None)
            
            if not video_track:
                self.logger.warning("No video track found in file", file_path=file_path)
                return {"is_hdr": False}
            
            hdr_metadata = {}
            
            # Check for HDR format based on transfer characteristics
            transfer_characteristics = None
            if hasattr(video_track, 'transfer_characteristics') and video_track.transfer_characteristics:
                transfer_characteristics = str(video_track.transfer_characteristics).lower()
                hdr_metadata["transfer_characteristics"] = transfer_characteristics
            
            # Determine HDR format
            is_hdr = False
            hdr_format = None
            hdr_format_commercial = ""
            
            # Check transfer characteristics for HDR indicators
            if transfer_characteristics:
                # Common HDR transfer characteristics
                if any(hdr_indicator in transfer_characteristics for hdr_indicator in [
                    'smpte2084', 'pq', 'perceptual quantizer',  # HDR10/HDR10+
                    'arib-std-b67', 'hlg', 'hybrid log-gamma',  # HLG
                    'smpte428', 'dci-p3'  # Cinema HDR
                ]):
                    is_hdr = True
                    
                    if 'smpte2084' in transfer_characteristics or 'pq' in transfer_characteristics:
                        hdr_format = "HDR10"
                    elif 'hlg' in transfer_characteristics or 'arib-std-b67' in transfer_characteristics:
                        hdr_format = "HLG"
                    elif 'smpte428' in transfer_characteristics:
                        hdr_format = "DCI-P3"
            
            # Check for commercial HDR format names
            if hasattr(video_track, 'hdr_format_commercial') and video_track.hdr_format_commercial:
                commercial_format = str(video_track.hdr_format_commercial).lower()
                hdr_format_commercial = commercial_format
                
                if not is_hdr:  # If not detected by transfer characteristics
                    if any(format_name in commercial_format for format_name in [
                        'hdr10+', 'hdr10 plus', 'dolby vision', 'hlg'
                    ]):
                        is_hdr = True
                        
                        if 'hdr10+' in commercial_format or 'hdr10 plus' in commercial_format:
                            hdr_format = "HDR10+"
                        elif 'dolby vision' in commercial_format:
                            hdr_format = "Dolby Vision"
                        elif 'hlg' in commercial_format:
                            hdr_format = "HLG"
            
            hdr_metadata.update({
                "is_hdr": is_hdr,
                "hdr_format": hdr_format,
                "hdr_format_commercial": hdr_format_commercial
            })
            
            # Extract color information
            color_info = self._extract_color_info(video_track)
            hdr_metadata.update(color_info)
            
            # Extract master display and content light level information
            display_info = self._extract_display_info(video_track)
            hdr_metadata.update(display_info)
            
            return hdr_metadata
            
        except Exception as e:
            self.logger.error("Failed to extract HDR metadata", exc_info=True, file_path=file_path)
            return {"is_hdr": False, "error": str(e)}
    
    def _extract_color_info(self, video_track) -> Dict[str, Any]:
        """Extract color space and related information"""
        color_info = {}
        
        # Color space
        if hasattr(video_track, 'color_space') and video_track.color_space:
            color_info["color_space"] = str(video_track.color_space)
        
        # Color primaries
        if hasattr(video_track, 'color_primaries') and video_track.color_primaries:
            color_info["color_primaries"] = str(video_track.color_primaries)
        
        # Matrix coefficients
        if hasattr(video_track, 'matrix_coefficients') and video_track.matrix_coefficients:
            color_info["matrix_coefficients"] = str(video_track.matrix_coefficients)
        
        # Color range
        if hasattr(video_track, 'color_range') and video_track.color_range:
            color_info["color_range"] = str(video_track.color_range)
        
        return color_info
    
    def _extract_display_info(self, video_track) -> Dict[str, Any]:
        """Extract master display and content light level information"""
        display_info = {}
        
        # Master display information (for HDR10)
        if hasattr(video_track, 'mastering_display_color_primaries') and video_track.mastering_display_color_primaries:
            display_info["master_display"] = str(video_track.mastering_display_color_primaries)
        
        # Content Light Level (CLL) - Maximum Content Light Level
        if hasattr(video_track, 'maximum_content_light_level') and video_track.maximum_content_light_level:
            try:
                display_info["max_cll"] = int(video_track.maximum_content_light_level)
            except (ValueError, TypeError):
                display_info["max_cll"] = str(video_track.maximum_content_light_level)
        
        # Frame Average Light Level (FALL) - Maximum Frame Average Light Level
        if hasattr(video_track, 'maximum_frameaverage_light_level') and video_track.maximum_frameaverage_light_level:
            try:
                display_info["max_fall"] = int(video_track.maximum_frameaverage_light_level)
            except (ValueError, TypeError):
                display_info["max_fall"] = str(video_track.maximum_frameaverage_light_level)
        
        return display_info
    
    async def _extract_codec_parameters(self, file_path: str) -> Dict[str, Any]:
        """Extract enhanced codec parameters that complement HDR metadata"""
        import pymediainfo # Import locally as it's an optional dependency
        
        try:
            media_info = pymediainfo.MediaInfo.parse(file_path)
            video_track = next((track for track in media_info.tracks if track.track_type == 'Video'), None)
            
            if not video_track:
                return {}
            
            codec_params = {}
            
            # Bit depth - crucial for HDR
            if hasattr(video_track, 'bit_depth') and video_track.bit_depth:
                try:
                    codec_params["bit_depth_video"] = int(video_track.bit_depth)
                except (ValueError, TypeError):
                    codec_params["bit_depth_video"] = str(video_track.bit_depth)
            
            # Enhanced format profile information
            if hasattr(video_track, 'format_profile') and video_track.format_profile:
                profile_info = str(video_track.format_profile)
                codec_params["format_profile_detailed"] = profile_info
                
                # Parse profile and level from format_profile
                if '@' in profile_info:
                    parts = profile_info.split('@')
                    if len(parts) >= 2:
                        codec_params["codec_profile"] = parts[0].strip()
                        level_part = parts[1].strip()
                        codec_params["codec_level"] = level_part
            
            # Chroma subsampling - important for color accuracy
            if hasattr(video_track, 'chroma_subsampling') and video_track.chroma_subsampling:
                codec_params["chroma_subsampling"] = str(video_track.chroma_subsampling)
            
            # Pixel format
            if hasattr(video_track, 'pixel_format') and video_track.pixel_format:
                codec_params["pixel_format"] = str(video_track.pixel_format)
            
            # Scan type and field order (important for interlaced content)
            if hasattr(video_track, 'scan_type') and video_track.scan_type:
                codec_params["scan_type"] = str(video_track.scan_type)
                
            if hasattr(video_track, 'scan_order') and video_track.scan_order: # Note: MediaInfo might use 'scan_order' or 'field_order'
                codec_params["field_order"] = str(video_track.scan_order)
            elif hasattr(video_track, 'field_order') and video_track.field_order:
                 codec_params["field_order"] = str(video_track.field_order)

            return codec_params
            
        except Exception as e:
            self.logger.error("Failed to extract codec parameters", exc_info=True, file_path=file_path)
            return {}
    
    async def setup(self):
        """Setup method called once when step is initialized"""
        if self.pymediainfo_available:
            self.logger.info("HDR Metadata Extractor step initialized.")
            self.logger.info("This step will attempt to extract HDR-specific metadata such as HDR format, color space, and mastering display information.")
        else:
            self.logger.info("HDR Metadata Extractor step is available but PyMediaInfo is not installed. HDR extraction will be skipped.")
    
    def get_info(self) -> Dict[str, Any]:
        """Get step information, including availability of PyMediaInfo"""
        info = super().get_info()
        info.update({
            "pymediainfo_available": self.pymediainfo_available,
            "supported_hdr_formats": ["HDR10", "HDR10+", "Dolby Vision", "HLG", "DCI-P3"],
            "color_metadata_extracted": [
                "color_space", "color_primaries", "matrix_coefficients", 
                "color_range", "transfer_characteristics"
            ],
            "hdr_specific_metadata_extracted": [
                "master_display", "max_cll", "max_fall", "bit_depth_video"
            ],
            "additional_codec_params_extracted": [
                "format_profile_detailed", "codec_profile", "codec_level",
                "chroma_subsampling", "pixel_format", "scan_type", "field_order"
            ]
        })
        return info