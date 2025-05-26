from typing import Dict, Any, List, Optional
import math
from pathlib import Path
from typing import Dict, Any, List, Optional # Added List, Optional for type hints
import cv2 # Conditionally imported in _check_opencv, but good to have here for clarity
import numpy as np # Conditionally imported with cv2, but good to have here for clarity

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata, StepConfig # Added StepConfig

class ExposureAnalysisStep(BaseStep):
    """Analyze exposure in video thumbnails using OpenCV"""

    name = "exposure_analysis"
    version = "1.0"
    description = "Analyze exposure quality in thumbnails - detect over/underexposure"
    category = "ai_analysis"

    requires = ["thumbnails"]
    provides = [
        "technical_metadata" # Exposure details will be nested here
    ]

    # Exposure analysis thresholds
    OVEREXPOSURE_THRESHOLD = 240  # Pixel values above this are considered overexposed
    UNDEREXPOSURE_THRESHOLD = 16  # Pixel values below this are considered underexposed
    WARNING_PERCENTAGE = 0.05     # 5% overexposed/underexposed pixels trigger warning

    def __init__(self, config: StepConfig): # Added type hint for config
        super().__init__(config)
        self.opencv_available = self._check_opencv()

    def _check_opencv(self) -> bool:
        """Check if OpenCV is available"""
        try:
            import cv2
            import numpy # numpy is often used with cv2
            self.logger.info("OpenCV and NumPy available for exposure analysis.")
            return True
        except ImportError:
            self.logger.warning("OpenCV or NumPy not available for exposure analysis.")
            self.logger.info("Install: pip install opencv-python numpy")
            return False

    async def process(self, video: VideoMetadata, context: Optional[Dict[str, Any]] = None) -> StepResult: # Added context
        """Analyze exposure in video thumbnails"""
        try:
            if not self.opencv_available:
                return StepResult(
                    success=False,
                    step_name=self.name,
                    video_id=video.video_id,
                    error="OpenCV not available for exposure analysis"
                )

            thumbnails = video.thumbnails
            if not thumbnails:
                return StepResult(
                    success=False,
                    step_name=self.name,
                    video_id=video.video_id,
                    error="No thumbnails available for exposure analysis"
                )

            # Analyze multiple thumbnails and average the results
            analyses: List[Dict[str, Any]] = [] # Added type hint

            for i, thumbnail_path in enumerate(thumbnails):
                if not Path(thumbnail_path).exists():
                    self.logger.warning(f"Thumbnail not found: {thumbnail_path}")
                    continue

                analysis = await self._analyze_single_thumbnail(thumbnail_path)
                if analysis:
                    analyses.append(analysis)

                    # Save partial results for each thumbnail analyzed
                    if self.save_partial: # Check if partial saving is enabled
                        await self.save_partial_result(
                            video.video_id,
                            video.user_id, # Pass user_id
                            {
                                f"thumbnail_{i}_exposure": analysis,
                                "exposure_analysis_progress": f"{i+1}/{len(thumbnails)} thumbnails analyzed"
                            }
                        )

            if not analyses:
                return StepResult(
                    success=False,
                    step_name=self.name,
                    video_id=video.video_id,
                    error="No thumbnails could be analyzed for exposure"
                )

            # Calculate overall exposure statistics
            overall_analysis = self._calculate_overall_exposure(analyses)

            # Determine exposure quality rating
            exposure_quality = self._rate_exposure_quality(overall_analysis)

            # Structure data for VideoExposureDetails model (to be nested in technical_metadata)
            exposure_details_data = {
                "warning": overall_analysis["warning"],
                "stops": overall_analysis["stops"],
                "overexposed_percentage": overall_analysis["overexposed_pct"],
                "underexposed_percentage": overall_analysis["underexposed_pct"],
                "overall_quality": exposure_quality, # Maps to VideoExposureDetails.overall_quality
                # Additional details not strictly in VideoExposureDetails but useful context
                "thumbnails_analyzed_count": len(analyses),
                "individual_thumbnail_analyses": analyses, # Store individual analyses if needed for debugging/details
                "avg_brightness_overall": overall_analysis["avg_brightness"],
                "brightness_std_overall": overall_analysis["brightness_std"]
            }
            
            # The main data to be saved by BaseStep should be structured for 'clips' table
            result_data = {
                "technical_metadata": {
                    "exposure_analysis": {k: v for k, v in exposure_details_data.items() if v is not None}
                }
            }
            
            # Log results
            warning_text = "⚠️ EXPOSURE WARNING" if overall_analysis["warning"] else "✓ Good exposure"
            self.logger.info(f"Exposure analysis complete: {warning_text}")
            self.logger.info(f"Overexposed: {overall_analysis['overexposed_pct']:.1f}%, Underexposed: {overall_analysis['underexposed_pct']:.1f}%")
            if overall_analysis["stops"] != 0:
                self.logger.info(f"Exposure deviation: {overall_analysis['stops']:+.1f} stops")

            return StepResult(
                success=True,
                step_name=self.name,
                video_id=video.video_id,
                data=result_data,
                metadata={
                    "thumbnails_analyzed_count": len(analyses), # Renamed for clarity
                    "analysis_method": "opencv_histogram",
                    "thresholds": {
                        "overexposure": self.OVEREXPOSURE_THRESHOLD,
                        "underexposure": self.UNDEREXPOSURE_THRESHOLD,
                        "warning_percentage": self.WARNING_PERCENTAGE
                    }
                }
            )

        except Exception as e:
            self.logger.error(f"Error in exposure analysis: {str(e)}", exc_info=True)
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video.video_id,
                error=str(e)
            )

    async def _analyze_single_thumbnail(self, thumbnail_path: str) -> Optional[Dict[str, Any]]: # Added Optional
        """
        Analyze exposure in a single thumbnail image.

        Args:
            thumbnail_path: Path to the thumbnail image

        Returns:
            Dict with exposure analysis results or None if analysis fails
        """
        try:
            # Ensure cv2 and numpy are imported for this method
            if not self.opencv_available: # Should have been checked in process, but good to be safe
                 self.logger.error("OpenCV not available for _analyze_single_thumbnail")
                 return None

            # Load image and convert to grayscale
            image = cv2.imread(thumbnail_path)
            if image is None:
                self.logger.warning(f"Could not load image: {thumbnail_path}")
                return None

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / (gray.shape[0] * gray.shape[1])  # Normalize to percentages

            # Calculate exposure statistics
            overexposed = np.sum(hist[self.OVEREXPOSURE_THRESHOLD:])
            underexposed = np.sum(hist[:self.UNDEREXPOSURE_THRESHOLD])

            # Calculate overall brightness statistics
            mean_brightness = np.mean(gray)
            brightness_std = np.std(gray)

            # Calculate exposure warning flag
            exposure_warning = overexposed > self.WARNING_PERCENTAGE or underexposed > self.WARNING_PERCENTAGE

            # Estimate exposure deviation in stops
            exposure_stops = self._calculate_exposure_stops(overexposed, underexposed, mean_brightness)

            return {
                "overexposed_pct": float(overexposed * 100),
                "underexposed_pct": float(underexposed * 100),
                "mean_brightness": float(mean_brightness),
                "brightness_std": float(brightness_std),
                "exposure_stops": exposure_stops,
                "warning": exposure_warning,
                "thumbnail_path": thumbnail_path
            }

        except Exception as e:
            self.logger.error(f"Failed to analyze thumbnail {thumbnail_path}: {str(e)}", exc_info=True)
            return None

    def _calculate_exposure_stops(self, overexposed_pct: float, underexposed_pct: float, mean_brightness: float) -> float:
        """
        Calculate exposure deviation in stops based on histogram analysis.

        Args:
            overexposed_pct: Percentage of overexposed pixels (0-1)
            underexposed_pct: Percentage of underexposed pixels (0-1)
            mean_brightness: Mean brightness value (0-255)

        Returns:
            Exposure deviation in stops (positive = overexposed, negative = underexposed)
        """
        # If significant overexposure, estimate positive stops
        if overexposed_pct > self.WARNING_PERCENTAGE:
            # Rough approximation: more overexposed pixels = more stops over
            stops = math.log2(max(overexposed_pct * 20, 1.1))  # Minimum 0.1 stops
            return min(stops, 5.0)  # Cap at 5 stops

        # If significant underexposure, estimate negative stops
        elif underexposed_pct > self.WARNING_PERCENTAGE:
            # Rough approximation: more underexposed pixels = more stops under
            stops = -math.log2(max(underexposed_pct * 20, 1.1))  # Minimum -0.1 stops
            return max(stops, -5.0)  # Cap at -5 stops

        # If no significant over/under exposure, use mean brightness to estimate deviation
        else:
            # Mean of 128 is "ideal" exposure for 8-bit image
            ideal_brightness = 128
            if ideal_brightness == 0: # Avoid division by zero
                return 0.0
            brightness_ratio = mean_brightness / ideal_brightness

            # Convert brightness ratio to stops (each stop is 2x brightness change)
            if brightness_ratio > 0 and brightness_ratio != 1.0: # math.log2 requires positive input
                stops = math.log2(brightness_ratio)
                # Only report significant deviations
                return stops if abs(stops) > 0.2 else 0.0

            return 0.0

    def _calculate_overall_exposure(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]: # Added type hint
        """Calculate overall exposure statistics from multiple thumbnail analyses"""
        if not analyses:
            return { # Return a default structure if no analyses
                "overexposed_pct": 0.0,
                "underexposed_pct": 0.0,
                "avg_brightness": 0.0,
                "brightness_std": 0.0,
                "stops": 0.0,
                "warning": False
            }

        # Average the exposure metrics
        avg_overexposed = np.mean([a["overexposed_pct"] for a in analyses])
        avg_underexposed = np.mean([a["underexposed_pct"] for a in analyses])
        avg_brightness = np.mean([a["mean_brightness"] for a in analyses])
        brightness_std = np.std([a["mean_brightness"] for a in analyses]) # This was correct

        # Calculate overall exposure stops (average, but weighted by severity)
        stops_values = [a["exposure_stops"] for a in analyses]
        avg_stops = np.mean(stops_values)

        # Overall warning if any thumbnail has significant issues OR average is problematic
        overall_warning = (
            avg_overexposed > self.WARNING_PERCENTAGE * 100 or
            avg_underexposed > self.WARNING_PERCENTAGE * 100 or
            any(a["warning"] for a in analyses)
        )

        return {
            "overexposed_pct": float(avg_overexposed),
            "underexposed_pct": float(avg_underexposed),
            "avg_brightness": float(avg_brightness),
            "brightness_std": float(brightness_std),
            "stops": float(avg_stops),
            "warning": overall_warning
        }

    def _rate_exposure_quality(self, analysis: Dict[str, Any]) -> str:
        """
        Rate overall exposure quality based on analysis.

        Returns:
            String rating: "excellent", "good", "fair", "poor"
        """
        if not analysis or not all(k in analysis for k in ["overexposed_pct", "underexposed_pct", "stops"]):
            return "unknown"

        overexposed = analysis["overexposed_pct"]
        underexposed = analysis["underexposed_pct"]
        stops = abs(analysis["stops"])

        # Excellent: minimal clipping, good exposure
        if overexposed < 1.0 and underexposed < 1.0 and stops < 0.5:
            return "excellent"

        # Good: some minor clipping or slight exposure deviation
        elif overexposed < 3.0 and underexposed < 3.0 and stops < 1.0:
            return "good"

        # Fair: noticeable exposure issues but still usable
        elif overexposed < 8.0 and underexposed < 8.0 and stops < 2.0:
            return "fair"

        # Poor: significant exposure problems
        else:
            return "poor"

    async def setup(self):
        """Setup method called once when step is initialized"""
        if self.opencv_available:
            self.logger.info("Exposure Analysis step initialized")
            self.logger.info(f"Thresholds - Overexposure: {self.OVEREXPOSURE_THRESHOLD}, Underexposure: {self.UNDEREXPOSURE_THRESHOLD}, Warning Pct: {self.WARNING_PERCENTAGE*100}%")
        else:
            self.logger.info("Exposure Analysis step disabled (OpenCV/NumPy unavailable)")

    def get_info(self) -> Dict[str, Any]:
        """Get step information"""
        info = super().get_info()
        info.update({
            "opencv_available": self.opencv_available,
            "thresholds": {
                "overexposure": self.OVEREXPOSURE_THRESHOLD,
                "underexposure": self.UNDEREXPOSURE_THRESHOLD,
                "warning_percentage": self.WARNING_PERCENTAGE
            },
            "quality_ratings": ["excellent", "good", "fair", "poor", "unknown"] # Added unknown
        })
        return info