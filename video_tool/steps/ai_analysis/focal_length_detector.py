from typing import Dict, Any, List, Optional, Tuple, Union # Added Union for _parse_shutter_speed
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from PIL import Image

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata, StepConfig

# Focal length category ranges (in mm, for full-frame equivalent)
FOCAL_LENGTH_RANGES = {
    "ULTRA-WIDE": (8, 18),    # Ultra wide-angle: 8-18mm
    "WIDE": (18, 35),         # Wide-angle: 18-35mm
    "MEDIUM": (35, 70),       # Standard/Normal: 35-70mm
    "LONG-LENS": (70, 200),   # Short telephoto: 70-200mm
    "TELEPHOTO": (200, 800)   # Telephoto: 200-800mm
}

class AIFocalLengthStep(BaseStep):
    """AI-powered focal length detection using Transformers when EXIF data unavailable"""

    name = "ai_focal_length"
    version = "1.0"
    description = "Detect focal length category using AI when EXIF data is missing"
    category = "ai_analysis"

    requires = ["thumbnails"]
    # This step contributes to the 'camera_details' JSONB field.
    provides = ["camera_details"]
    optional_requires = ["focal_length_mm", "focal_length_category"]  # From EXIF

    def __init__(self, config: StepConfig):
        super().__init__(config)
        self.transformers_available = False
        self.device = "cpu"
        self.pipeline = None

        # Check if we should run AI detection
        self.enabled = config.params.get("enabled", False)

        if self.enabled:
            self._check_dependencies()

    def _check_dependencies(self):
        """Check if required AI libraries are available"""
        try:
            from transformers import pipeline
            import torch

            self.transformers_available = True

            # Device selection logic - prioritize MPS, then CUDA, then CPU
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                self.logger.info("Using MPS (Apple Silicon) acceleration")
            elif torch.cuda.is_available():
                self.device = "cuda"
                self.logger.info("Using CUDA acceleration")
            else:
                self.device = "cpu"
                self.logger.info("Using CPU for AI focal length detection")

        except ImportError as e:
            self.logger.warning(f"AI focal length detection disabled: {str(e)}")
            self.logger.info("Install: pip install transformers torch pillow")
            self.transformers_available = False

    async def process(self, video: VideoMetadata, context: Optional[Dict[str, Any]] = None) -> StepResult:
        """Process focal length detection"""
        try:
            # Check if we already have focal length from EXIF
            if self._has_exif_focal_length(video):
                self.logger.info("EXIF focal length available, skipping AI detection")
                return StepResult(
                    success=True,
                    step_name=self.name,
                    video_id=video.video_id,
                    data={
                        "camera_details": { # Nest under camera_details
                            "focal_length": {
                                "source": "EXIF",
                                # If EXIF already provided category/mm, they'd be in video object.
                                # This step just confirms EXIF was the source.
                                # No need to explicitly set category/mm to None here if EXIF provided them.
                            }
                        }
                    }
                )

            # Check if AI detection is enabled and available
            if not self.enabled or not self.transformers_available:
                self.logger.info("AI focal length detection disabled or unavailable")
                return StepResult(
                    success=True,
                    step_name=self.name,
                    video_id=video.video_id,
                    data={
                         "camera_details": { # Nest under camera_details
                            "focal_length": {
                                "source": "unavailable",
                                "category": None, # Explicitly set to None if unavailable
                                "value_mm": None,
                                "ai_confidence": None
                            }
                        }
                    }
                )

            # Get best thumbnail for analysis
            thumbnails = video.thumbnails
            if not thumbnails:
                return StepResult(
                    success=False,
                    step_name=self.name,
                    video_id=video.video_id,
                    error="No thumbnails available for AI focal length detection"
                )

            # Use the middle thumbnail (usually most representative)
            middle_idx = len(thumbnails) // 2
            thumbnail_path = thumbnails[middle_idx]

            # Run AI detection
            category, confidence = await self._detect_focal_length_ai(thumbnail_path)

            # Data structured for CameraFocalLength model, nested under camera_details
            result_data = {
                "camera_details": {
                    "focal_length": {
                        "source": "AI",
                        "category": category,
                        "ai_confidence": confidence, # Store AI confidence if applicable
                        "value_mm": None  # AI doesn't provide exact mm
                    }
                }
            }
            
            # Filter None values from the focal_length part
            if result_data["camera_details"]["focal_length"].get("category") is None:
                 del result_data["camera_details"]["focal_length"]["category"]
            if result_data["camera_details"]["focal_length"].get("ai_confidence") is None:
                 del result_data["camera_details"]["focal_length"]["ai_confidence"]


            self.logger.info(f"AI detected focal length: {category} (confidence: {confidence:.3f})")

            return StepResult(
                success=True,
                step_name=self.name,
                video_id=video.video_id,
                data=result_data,
                metadata={
                    "model": "tonyassi/camera-lens-focal-length",
                    "device": self.device,
                    "thumbnail_used": thumbnail_path
                }
            )

        except Exception as e:
            self.logger.error(f"Error in AI Focal Length Step: {str(e)}", exc_info=True)
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video.video_id,
                error=str(e)
            )

    def _has_exif_focal_length(self, video: VideoMetadata) -> bool:
        """Check if video already has focal length from EXIF data"""
        # Check if we have focal length data from previous EXIF extraction steps
        # This would be populated by metadata extraction steps
        return (
            hasattr(video, 'focal_length_mm') and video.focal_length_mm is not None
        ) or (
            hasattr(video, 'focal_length_category') and video.focal_length_category is not None
        )

    async def _detect_focal_length_ai(self, image_path: str) -> Tuple[Optional[str], float]:
        """
        Use AI to detect the focal length category from an image.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (category, confidence)
        """
        try:
            # Import here to avoid errors if not available
            from transformers import pipeline

            # Initialize pipeline if not already done
            if self.pipeline is None:
                self.logger.info("Loading AI focal length detection model...")
                self.pipeline = pipeline(
                    "image-classification",
                    model="tonyassi/camera-lens-focal-length",
                    device=self.device
                )
                self.logger.info("Model loaded successfully")

            # Load and process image
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Thumbnail not found: {image_path}")

            pil_image = Image.open(image_path)

            # Run the model to estimate focal length category
            self.logger.debug(f"Running AI detection on {image_path}")
            prediction_result = self.pipeline(pil_image)

            # Extract the top prediction
            if prediction_result and len(prediction_result) > 0:
                top_prediction = prediction_result[0]
                category = top_prediction["label"]
                confidence = top_prediction["score"]

                # Map model output to our standard categories if needed
                mapped_category = self._map_ai_category(category)

                return mapped_category, confidence
            else:
                self.logger.warning("AI model returned no predictions")
                return None, 0.0

        except Exception as e:
            self.logger.error(f"AI focal length detection failed: {str(e)}", exc_info=True)
            return None, 0.0

    def _map_ai_category(self, ai_category: str) -> str:
        """
        Map AI model output to our standard focal length categories.

        The tonyassi/camera-lens-focal-length model may use different category names,
        so we normalize them to our standard FOCAL_LENGTH_RANGES keys.
        """
        # Convert to uppercase and handle common variations
        ai_category_normalized = ai_category.upper().replace("-", "_").replace(" ", "_")

        # Direct mapping
        if ai_category_normalized in FOCAL_LENGTH_RANGES:
            return ai_category_normalized

        # Handle variations
        category_mappings = {
            "ULTRAWIDE": "ULTRA-WIDE",
            "ULTRA_WIDE": "ULTRA-WIDE",
            "WIDE_ANGLE": "WIDE",
            "NORMAL": "MEDIUM",
            "STANDARD": "MEDIUM",
            # "TELEPHOTO" is already a key in FOCAL_LENGTH_RANGES
            "TELE": "TELEPHOTO", # if model outputs just "TELE"
            "LONG": "LONG-LENS" # if model outputs just "LONG"
        }

        mapped = category_mappings.get(ai_category_normalized, ai_category_normalized)

        # Final validation - return valid category or default to MEDIUM
        if mapped in FOCAL_LENGTH_RANGES:
            return mapped
        else:
            self.logger.warning(f"Unknown AI category '{ai_category}' (normalized to '{ai_category_normalized}'), defaulting to MEDIUM")
            return "MEDIUM"

    async def setup(self):
        """Setup method called once when step is initialized"""
        if self.enabled and self.transformers_available:
            self.logger.info("AI Focal Length Detection step initialized")
            self.logger.info(f"Device: {self.device}")
            self.logger.info("Model will be loaded on first use")
        else:
            reason = "disabled" if not self.enabled else "dependencies unavailable"
            self.logger.info(f"AI Focal Length Detection step skipped ({reason})")

    def get_info(self) -> Dict[str, Any]:
        """Get step information"""
        info = super().get_info()
        info.update({
            "enabled": self.enabled,
            "transformers_available": self.transformers_available,
            "device": self.device,
            "model": "tonyassi/camera-lens-focal-length",
            "categories": list(FOCAL_LENGTH_RANGES.keys())
        })
        return info