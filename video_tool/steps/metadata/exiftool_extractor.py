from pathlib import Path
from typing import Dict, Any, Optional, Union
import structlog
from datetime import datetime
import dateutil.parser as dateutil_parser

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata

# Focal length category ranges (in mm, for full-frame equivalent)
FOCAL_LENGTH_RANGES = {
    "ULTRA-WIDE": (8, 18),    # Ultra wide-angle: 8-18mm
    "WIDE": (18, 35),         # Wide-angle: 18-35mm
    "MEDIUM": (35, 70),       # Standard/Normal: 35-70mm
    "LONG-LENS": (70, 200),   # Short telephoto: 70-200mm
    "TELEPHOTO": (200, 800)   # Telephoto: 200-800mm
}

class ExifToolExtractorStep(BaseStep):
    """Extract comprehensive EXIF metadata using PyExifTool"""

    name = "exiftool_extractor"
    version = "1.0"
    description = "Extract comprehensive EXIF metadata including camera settings and GPS"
    category = "metadata"

    requires = ["file_path"]
    provides = [
        # Camera information
        "camera_make", "camera_model", "lens_model", "camera_serial_number",
        # Shooting settings
        "focal_length_mm", "focal_length_category", "iso", "shutter_speed",
        "f_stop", "exposure_mode", "white_balance",
        # GPS and location
        "gps_latitude", "gps_longitude", "gps_altitude", "location_name",
        # Timestamps
        "date_taken", "date_created", "date_modified",
        # Additional metadata
        "software", "artist", "copyright", "description"
    ]

    def __init__(self, config):
        super().__init__(config)
        self.exiftool_available = self._check_exiftool()

    def _check_exiftool(self) -> bool:
        """Check if PyExifTool and ExifTool are available"""
        try:
            import exiftool
            # Try to create a basic ExifTool instance to verify it works
            with exiftool.ExifTool() as et:
                pass  # Just test that it can start
            return True
        except ImportError:
            self.logger.warning("PyExifTool not available for EXIF metadata extraction")
            self.logger.info("Install: pip install PyExifTool")
            return False
        except Exception as e:
            self.logger.warning(f"ExifTool not available: {str(e)}")
            self.logger.info("Install ExifTool from https://exiftool.org/")
            return False

    async def process(self, video: VideoMetadata) -> StepResult:
        """Extract EXIF metadata from video file"""
        try:
            if not self.exiftool_available:
                return StepResult(
                    success=False,
                    step_name=self.name,
                    video_id=video.video_id,
                    error="ExifTool not available for EXIF metadata extraction"
                )

            file_path = video.file_path

            if not Path(file_path).exists():
                return StepResult(
                    success=False,
                    step_name=self.name,
                    video_id=video.video_id,
                    error=f"File not found: {file_path}"
                )

            # Extract EXIF metadata
            exif_metadata = await self._extract_exif_metadata(file_path)

            # Process and enhance the metadata
            processed_metadata = self._process_exif_data(exif_metadata)

            # Log important findings
            camera_info = []
            if processed_metadata.get("camera_make"):
                camera_info.append(processed_metadata["camera_make"])
            if processed_metadata.get("camera_model"):
                camera_info.append(processed_metadata["camera_model"])

            if camera_info:
                self.logger.info(f"Camera detected: {' '.join(camera_info)}")

            if processed_metadata.get("focal_length_mm"):
                focal_length = processed_metadata["focal_length_mm"]
                category = processed_metadata.get("focal_length_category", "Unknown")
                self.logger.info(f"Focal length: {focal_length}mm ({category})")

            return StepResult(
                success=True,
                step_name=self.name,
                video_id=video.video_id,
                data=processed_metadata,
                metadata={
                    "extraction_method": "exiftool",
                    "has_gps": bool(processed_metadata.get("gps_latitude")),
                    "has_camera_settings": bool(processed_metadata.get("iso"))
                }
            )

        except Exception as e:
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video.video_id,
                error=str(e)
            )

    async def _extract_exif_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract EXIF metadata using PyExifTool.

        Args:
            file_path: Path to the video file

        Returns:
            Dict with raw EXIF metadata
        """
        import exiftool

        try:
            with exiftool.ExifTool() as et:
                metadata = et.get_metadata(file_path)[0]
                return metadata

        except Exception as e:
            self.logger.error(f"Failed to extract EXIF metadata: {str(e)}")
            return {}

    def _process_exif_data(self, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and clean the raw EXIF metadata into standardized fields.

        Args:
            raw_metadata: Raw metadata from ExifTool

        Returns:
            Dict with processed and standardized metadata
        """
        processed = {} # For direct columns in 'clips'
        camera_details_data = {} # For 'camera_details' JSONB
        technical_metadata_data = {} # For 'technical_metadata' JSONB

        # Camera information (direct columns)
        processed["camera_make"] = raw_metadata.get('EXIF:Make') or raw_metadata.get('QuickTime:Make')
        processed["camera_model"] = raw_metadata.get('EXIF:Model') or raw_metadata.get('QuickTime:Model')
        
        # Camera information (for camera_details JSONB)
        camera_details_data["lens_model"] = raw_metadata.get('EXIF:LensModel') or raw_metadata.get('QuickTime:LensModel')
        camera_details_data["camera_serial_number"] = raw_metadata.get('EXIF:SerialNumber') or raw_metadata.get('QuickTime:SerialNumber')

        # Software information (for technical_metadata JSONB)
        technical_metadata_data["software"] = raw_metadata.get('EXIF:Software') or raw_metadata.get('QuickTime:Software')

        # Focal length processing (for camera_details JSONB)
        focal_length_raw = raw_metadata.get('EXIF:FocalLength') or raw_metadata.get('QuickTime:FocalLength')
        if focal_length_raw:
            focal_length_mm = self._parse_focal_length(focal_length_raw)
            if focal_length_mm:
                camera_details_data["focal_length_mm"] = focal_length_mm
                camera_details_data["focal_length_category"] = self._categorize_focal_length(focal_length_mm)
                camera_details_data["focal_length_source"] = "EXIF"

        # Camera settings (for camera_details JSONB)
        camera_details_data["iso"] = self._parse_numeric(raw_metadata.get('EXIF:ISO') or raw_metadata.get('QuickTime:ISO'))
        camera_details_data["shutter_speed"] = self._parse_shutter_speed(
            raw_metadata.get('EXIF:ShutterSpeedValue') or
            raw_metadata.get('EXIF:ExposureTime') or
            raw_metadata.get('QuickTime:ExposureTime')
        )
        camera_details_data["f_stop"] = self._parse_numeric(
            raw_metadata.get('EXIF:FNumber') or
            raw_metadata.get('EXIF:ApertureValue') or
            raw_metadata.get('QuickTime:Aperture')
        )

        # Exposure and white balance (for camera_details JSONB)
        camera_details_data["exposure_mode"] = self._map_exposure_mode(raw_metadata.get('EXIF:ExposureMode'))
        camera_details_data["white_balance"] = self._map_white_balance(raw_metadata.get('EXIF:WhiteBalance'))

        # GPS information (for camera_details JSONB)
        gps_lat = raw_metadata.get('EXIF:GPSLatitude') or raw_metadata.get('QuickTime:GPSLatitude')
        gps_lon = raw_metadata.get('EXIF:GPSLongitude') or raw_metadata.get('QuickTime:GPSLongitude')
        gps_alt = raw_metadata.get('EXIF:GPSAltitude') or raw_metadata.get('QuickTime:GPSAltitude')

        if gps_lat:
            camera_details_data["gps_latitude"] = self._parse_gps_coordinate(gps_lat)
        if gps_lon:
            camera_details_data["gps_longitude"] = self._parse_gps_coordinate(gps_lon)
        if gps_alt:
            camera_details_data["gps_altitude"] = self._parse_numeric(gps_alt)
        
        camera_details_data["location_name"] = self._build_location_name(raw_metadata)


        # Timestamps
        # 'created_at' in 'clips' table is usually file system creation or first processing.
        # 'date_taken' is more specific to when the photo/video was shot.
        technical_metadata_data["date_taken"] = self._parse_datetime(
            raw_metadata.get('EXIF:DateTimeOriginal') or
            raw_metadata.get('EXIF:CreateDate') or
            raw_metadata.get('QuickTime:CreateDate')
        )
        # File system dates might be useful in technical_metadata if different from DB created_at
        technical_metadata_data["file_system_created_at"] = self._parse_datetime(raw_metadata.get('File:FileCreateDate'))
        technical_metadata_data["file_system_modified_at"] = self._parse_datetime(raw_metadata.get('File:FileModifyDate'))


        # Additional metadata (for technical_metadata JSONB)
        technical_metadata_data["artist"] = raw_metadata.get('EXIF:Artist') or raw_metadata.get('QuickTime:Artist')
        technical_metadata_data["copyright"] = raw_metadata.get('EXIF:Copyright') or raw_metadata.get('QuickTime:Copyright')
        
        # 'description' from EXIF might be different from AI-generated content_summary
        technical_metadata_data["exif_description"] = raw_metadata.get('EXIF:ImageDescription') or raw_metadata.get('QuickTime:Description')

        # Add the structured JSONB data to the main processed dictionary
        if camera_details_data:
            processed["camera_details"] = {k: v for k, v in camera_details_data.items() if v is not None}
        if technical_metadata_data:
            processed["technical_metadata"] = {k: v for k, v in technical_metadata_data.items() if v is not None}
            
        # Filter out None values from the top-level processed dictionary
        return {k: v for k, v in processed.items() if v is not None}

    def _parse_focal_length(self, focal_length_raw: Any) -> Optional[float]:
        """Parse focal length from various EXIF formats"""
        if not focal_length_raw:
            return None

        try:
            # Handle string formats like "24.0 mm" or "24mm"
            if isinstance(focal_length_raw, str):
                focal_length_str = focal_length_raw.lower().replace('mm', '').strip()
                return float(focal_length_str)

            # Handle numeric values
            return float(focal_length_raw)

        except (ValueError, TypeError):
            self.logger.warning(f"Could not parse focal length: {focal_length_raw}")
            return None

    def _categorize_focal_length(self, focal_mm: float) -> str:
        """Categorize focal length into standard ranges"""
        for category, (min_val, max_val) in FOCAL_LENGTH_RANGES.items():
            if min_val <= focal_mm < max_val:
                return category

        # Handle edge cases
        if focal_mm < 8:
            return "ULTRA-WIDE"
        elif focal_mm >= 800:
            return "TELEPHOTO"
        else:
            return "MEDIUM"  # Default fallback

    def _parse_numeric(self, value: Any) -> Optional[Union[int, float]]:
        """Parse numeric values from EXIF data"""
        if value is None:
            return None

        try:
            # Try integer first
            if isinstance(value, str) and '.' not in value:
                return int(value)
            # Then float
            return float(value)
        except (ValueError, TypeError):
            return None

    def _parse_shutter_speed(self, value: Any) -> Optional[Union[str, float]]:
        """Parse shutter speed from various EXIF formats"""
        if not value:
            return None

        try:
            # If it's already a number, use it
            if isinstance(value, (int, float)):
                if value >= 1:
                    return f"{value}s"
                else:
                    return f"1/{int(1/value)}"

            # If it's a string, try to parse it
            if isinstance(value, str):
                # Handle formats like "1/60" or "0.0167"
                if '/' in value:
                    return value
                else:
                    speed = float(value)
                    if speed >= 1:
                        return f"{speed}s"
                    else:
                        return f"1/{int(1/speed)}"

        except (ValueError, TypeError):
            self.logger.warning(f"Could not parse shutter speed: {value}")

        return None

    def _map_exposure_mode(self, mode_val: Any) -> Optional[str]:
        """Map EXIF exposure mode values to readable strings"""
        if not mode_val:
            return None

        try:
            val = int(str(mode_val).strip())
            exposure_modes = {
                0: "Auto",
                1: "Manual",
                2: "Auto bracket"
            }
            return exposure_modes.get(val, f"Unknown ({val})")
        except (ValueError, TypeError):
            return str(mode_val) if mode_val else None

    def _map_white_balance(self, wb_val: Any) -> Optional[str]:
        """Map EXIF white balance values to readable strings"""
        if not wb_val:
            return None

        try:
            val = int(str(wb_val).strip())
            white_balance_modes = {
                0: "Auto",
                1: "Daylight",
                2: "Fluorescent",
                3: "Tungsten",
                4: "Flash",
                9: "Fine weather",
                10: "Cloudy",
                11: "Shade"
            }
            return white_balance_modes.get(val, f"Unknown ({val})")
        except (ValueError, TypeError):
            return str(wb_val) if wb_val else None

    def _parse_gps_coordinate(self, coord: Any) -> Optional[float]:
        """Parse GPS coordinates from EXIF format"""
        if not coord:
            return None

        try:
            # GPS coordinates are often in decimal degrees already
            if isinstance(coord, (int, float)):
                return float(coord)

            # Handle string formats
            if isinstance(coord, str):
                # Remove direction indicators (N, S, E, W) if present
                coord_clean = coord.replace('N', '').replace('S', '').replace('E', '').replace('W', '').strip()
                return float(coord_clean)

        except (ValueError, TypeError):
            self.logger.warning(f"Could not parse GPS coordinate: {coord}")

        return None

    def _build_location_name(self, raw_metadata: Dict[str, Any]) -> Optional[str]:
        """Build location name from available EXIF location fields"""
        location_parts = []

        # Try various location fields
        city = raw_metadata.get('IPTC:City') or raw_metadata.get('XMP:City')
        state = raw_metadata.get('IPTC:Province-State') or raw_metadata.get('XMP:State')
        country = raw_metadata.get('IPTC:Country-PrimaryLocationName') or raw_metadata.get('XMP:Country')

        if city:
            location_parts.append(city)
        if state and state != city:
            location_parts.append(state)
        if country and country not in location_parts:
            location_parts.append(country)

        return ', '.join(location_parts) if location_parts else None

    def _parse_datetime(self, datetime_str: Any) -> Optional[datetime]:
        """Parse datetime strings from EXIF data"""
        if not datetime_str:
            return None

        try:
            # Handle various datetime formats
            if isinstance(datetime_str, str):
                # Clean up common EXIF datetime issues
                cleaned_date_str = datetime_str.strip()

                # Replace timezone indicators
                cleaned_date_str = cleaned_date_str.replace(' UTC', 'Z')

                # Fix malformed time parts (sometimes there are dashes instead of colons)
                if ' ' in cleaned_date_str:
                    date_part, time_part = cleaned_date_str.split(' ', 1)
                    if '-' in time_part and ':' not in time_part:
                        time_part = time_part.replace('-', ':')
                        cleaned_date_str = f"{date_part} {time_part}"

                # Try to parse with dateutil
                return dateutil_parser.parse(cleaned_date_str)

        except Exception as e:
            self.logger.warning(f"Could not parse datetime '{datetime_str}': {str(e)}")

        return None

    async def setup(self):
        """Setup method called once when step is initialized"""
        if self.exiftool_available:
            self.logger.info("ExifTool Extractor step initialized")
            self.logger.info("Extracting camera settings, GPS, and comprehensive metadata")
        else:
            self.logger.info("ExifTool Extractor step disabled (ExifTool unavailable)")

    def get_info(self) -> Dict[str, Any]:
        """Get step information"""
        info = super().get_info()
        info.update({
            "exiftool_available": self.exiftool_available,
            "focal_length_categories": list(FOCAL_LENGTH_RANGES.keys()),
            "metadata_types": [
                "camera_information", "shooting_settings", "gps_location",
                "timestamps", "copyright_info"
            ]
        })
        return info