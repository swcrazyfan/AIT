from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import uuid

# =====================================================
# AUDIO MODELS
# =====================================================

class AudioTrack(BaseModel):
    """Detailed audio track information"""
    track_id: Optional[str] = None
    codec: Optional[str] = None
    codec_id: Optional[str] = None
    channels: Optional[int] = None
    channel_layout: Optional[str] = None
    sample_rate: Optional[int] = None
    bit_depth: Optional[int] = None
    bit_rate_kbps: Optional[int] = None
    language: Optional[str] = None
    duration_seconds: Optional[float] = None

class SubtitleTrack(BaseModel):
    """Subtitle track information"""
    track_id: Optional[str] = None
    codec: Optional[str] = None
    format: Optional[str] = None
    language: Optional[str] = None
    title: Optional[str] = None
    embedded: Optional[bool] = None

# =====================================================
# VIDEO TECHNICAL MODELS
# =====================================================

class VideoCodecDetails(BaseModel):
    """Detailed video codec information"""
    name: Optional[str] = None
    profile: Optional[str] = None
    level: Optional[str] = None
    bitrate_kbps: Optional[int] = None
    bit_depth: Optional[int] = None
    chroma_subsampling: Optional[str] = None
    pixel_format: Optional[str] = None
    bitrate_mode: Optional[str] = None  # CBR, VBR, etc.
    cabac: Optional[bool] = None  # H.264 specific
    ref_frames: Optional[int] = None
    gop_size: Optional[int] = None
    scan_type: Optional[str] = None  # Progressive, Interlaced
    field_order: Optional[str] = None

class VideoResolution(BaseModel):
    """Video resolution information"""
    width: Optional[int] = None
    height: Optional[int] = None
    aspect_ratio: Optional[str] = None  # "16:9", "4:3", etc.

class VideoHDRDetails(BaseModel):
    """HDR-specific metadata"""
    is_hdr: bool = False
    hdr_format: Optional[str] = None  # HDR10, HDR10+, Dolby Vision, HLG
    hdr_format_commercial: Optional[str] = None
    transfer_characteristics: Optional[str] = None
    master_display: Optional[str] = None
    max_cll: Optional[int] = None  # Maximum Content Light Level
    max_fall: Optional[int] = None  # Maximum Frame Average Light Level

class VideoColorDetails(BaseModel):
    """Color space and characteristics"""
    color_space: Optional[str] = None
    color_primaries: Optional[str] = None
    transfer_characteristics: Optional[str] = None
    matrix_coefficients: Optional[str] = None
    color_range: Optional[str] = None  # Limited, Full
    hdr: VideoHDRDetails = Field(default_factory=VideoHDRDetails)

class VideoExposureDetails(BaseModel):
    """Exposure analysis results"""
    warning: Optional[bool] = None
    stops: Optional[float] = None  # Exposure deviation in stops
    overexposed_percentage: Optional[float] = None
    underexposed_percentage: Optional[float] = None
    overall_quality: Optional[str] = None  # excellent, good, fair, poor

class VideoDetails(BaseModel):
    """Complete video technical metadata"""
    codec: VideoCodecDetails = Field(default_factory=VideoCodecDetails)
    container: Optional[str] = None
    resolution: VideoResolution = Field(default_factory=VideoResolution)
    frame_rate: Optional[float] = None
    color: VideoColorDetails = Field(default_factory=VideoColorDetails)
    exposure: VideoExposureDetails = Field(default_factory=VideoExposureDetails)

# =====================================================
# CAMERA MODELS
# =====================================================

class CameraFocalLength(BaseModel):
    """Focal length information"""
    value_mm: Optional[float] = None
    category: Optional[str] = None  # ULTRA-WIDE, WIDE, MEDIUM, LONG-LENS, TELEPHOTO
    source: Optional[str] = None  # EXIF, AI, Manual

class CameraSettings(BaseModel):
    """Camera shooting settings"""
    iso: Optional[int] = None
    shutter_speed: Optional[Union[str, float]] = None
    f_stop: Optional[float] = None
    exposure_mode: Optional[str] = None
    white_balance: Optional[str] = None

class CameraLocation(BaseModel):
    """GPS and location information"""
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude: Optional[float] = None
    location_name: Optional[str] = None

class CameraDetails(BaseModel):
    """Complete camera and lens information"""
    make: Optional[str] = None
    model: Optional[str] = None
    lens_model: Optional[str] = None
    serial_number: Optional[str] = None
    focal_length: CameraFocalLength = Field(default_factory=CameraFocalLength)
    settings: CameraSettings = Field(default_factory=CameraSettings)
    location: CameraLocation = Field(default_factory=CameraLocation)

# =====================================================
# AI ANALYSIS MODELS
# =====================================================

class ShotType(BaseModel):
    """Individual shot type detection"""
    timestamp: str
    shot_type: str  # Close-up, Medium shot, Wide shot, etc.
    description: str
    confidence: Optional[float] = None

class TechnicalQuality(BaseModel):
    """Technical quality assessment"""
    overall_focus_quality: Optional[str] = None
    stability_assessment: Optional[str] = None
    detected_artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    usability_rating: Optional[str] = None

class DetectedText(BaseModel):
    """Text detection results"""
    text_content: Optional[str] = None
    text_type: Optional[str] = None  # Title, Subtitle, Caption, etc.
    readability: Optional[str] = None

class DetectedLogo(BaseModel):
    """Logo/icon detection results"""
    element_type: str
    description: Optional[str] = None
    size: Optional[str] = None

class TextAndGraphics(BaseModel):
    """Text and graphics analysis"""
    detected_text: List[DetectedText] = Field(default_factory=list)
    detected_logos_icons: List[DetectedLogo] = Field(default_factory=list)

class RecommendedKeyframe(BaseModel):
    """Keyframe recommendation"""
    timestamp: str
    reason: str
    visual_quality: str

class KeyframeAnalysis(BaseModel):
    """Keyframe analysis results"""
    recommended_keyframes: List[RecommendedKeyframe] = Field(default_factory=list)

class VisualAnalysis(BaseModel):
    """Complete visual analysis results"""
    shot_types: List[ShotType] = Field(default_factory=list)
    technical_quality: Optional[TechnicalQuality] = None
    text_and_graphics: Optional[TextAndGraphics] = None
    keyframe_analysis: Optional[KeyframeAnalysis] = None

# =====================================================
# AUDIO ANALYSIS MODELS
# =====================================================

class TranscriptSegment(BaseModel):
    """Individual transcript segment"""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    speaker: Optional[str] = None
    text: str
    confidence: Optional[float] = None

class Transcript(BaseModel):
    """Complete transcript information"""
    full_text: Optional[str] = None
    segments: List[TranscriptSegment] = Field(default_factory=list)
    word_count: Optional[int] = None
    language: Optional[str] = None

class Speaker(BaseModel):
    """Speaker identification"""
    speaker_id: str
    speaking_time_seconds: float
    segments_count: Optional[int] = None
    confidence: Optional[float] = None

class SpeakerAnalysis(BaseModel):
    """Speaker analysis results"""
    speaker_count: int = 0
    speakers: List[Speaker] = Field(default_factory=list)

class SoundEvent(BaseModel):
    """Detected sound event"""
    timestamp: Optional[str] = None
    event_type: str
    description: Optional[str] = None
    prominence: Optional[str] = None  # High, Medium, Low
    confidence: Optional[float] = None

class AudioQuality(BaseModel):
    """Audio quality assessment"""
    clarity: Optional[str] = None
    background_noise_level: Optional[str] = None
    dialogue_intelligibility: Optional[str] = None
    overall_rating: Optional[str] = None

class AudioAnalysis(BaseModel):
    """Complete audio analysis results"""
    transcript: Optional[Transcript] = None
    speaker_analysis: Optional[SpeakerAnalysis] = None
    sound_events: List[SoundEvent] = Field(default_factory=list)
    audio_quality: Optional[AudioQuality] = None

# =====================================================
# CONTENT ANALYSIS MODELS
# =====================================================

class PersonDetail(BaseModel):
    """Person detection details"""
    person_id: Optional[str] = None
    role: Optional[str] = None
    visibility_duration: Optional[str] = None
    confidence: Optional[float] = None

class Location(BaseModel):
    """Location detection"""
    name: str
    type: str  # Indoor, Outdoor, Studio, etc.
    description: Optional[str] = None
    confidence: Optional[float] = None

class ObjectOfInterest(BaseModel):
    """Object detection"""
    object: str
    significance: str
    timestamp: Optional[str] = None
    confidence: Optional[float] = None

class Entities(BaseModel):
    """Entity detection results"""
    people_count: int = 0
    people_details: List[PersonDetail] = Field(default_factory=list)
    locations: List[Location] = Field(default_factory=list)
    objects_of_interest: List[ObjectOfInterest] = Field(default_factory=list)

class Activity(BaseModel):
    """Activity detection"""
    activity: str
    timestamp: Optional[str] = None
    duration: Optional[str] = None
    importance: str  # High, Medium, Low
    confidence: Optional[float] = None

class ContentWarning(BaseModel):
    """Content warning detection"""
    type: str
    severity: str
    description: Optional[str] = None
    timestamp: Optional[str] = None

class ContentAnalysis(BaseModel):
    """Complete content analysis results"""
    entities: Optional[Entities] = None
    activity_summary: List[Activity] = Field(default_factory=list)
    content_warnings: List[ContentWarning] = Field(default_factory=list)

# =====================================================
# AI ANALYSIS SUMMARY
# =====================================================

class AIAnalysisSummary(BaseModel):
    """High-level AI analysis summary"""
    overall: Optional[str] = None
    key_activities: List[str] = Field(default_factory=list)
    content_category: Optional[str] = None
    mood_tone: Optional[str] = None
    primary_subjects: List[str] = Field(default_factory=list)

class ComprehensiveAIAnalysis(BaseModel):
    """Complete AI analysis container"""
    summary: Optional[AIAnalysisSummary] = None
    visual_analysis: Optional[VisualAnalysis] = None
    audio_analysis: Optional[AudioAnalysis] = None
    content_analysis: Optional[ContentAnalysis] = None
    analysis_file_path: Optional[str] = None
    ai_model: Optional[str] = None
    processing_time_seconds: Optional[float] = None

# =====================================================
# FILE INFORMATION MODELS
# =====================================================

class FileInfo(BaseModel):
    """Complete file information"""
    file_path: str
    file_name: str
    file_checksum: str
    file_size_bytes: int
    created_at: Optional[datetime] = None
    processed_at: datetime = Field(default_factory=datetime.now)
    is_duplicate: Optional[bool] = None
    duplicate_of: Optional[str] = None

# =====================================================
# ANALYSIS DETAILS CONTAINER
# =====================================================

class AnalysisDetails(BaseModel):
    """Container for analysis results"""
    scene_changes: List[float] = Field(default_factory=list)
    content_tags: List[str] = Field(default_factory=list)
    content_summary: Optional[str] = None
    ai_analysis: Optional[ComprehensiveAIAnalysis] = None
    processing_metadata: Optional[Dict[str, Any]] = None

# =====================================================
# MAIN OUTPUT MODEL
# =====================================================

class EnhancedVideoIngestOutput(BaseModel):
    """Enhanced complete video ingest output with all detailed models"""

    # Core identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None

    # File information
    file_info: FileInfo

    # Technical metadata
    video: VideoDetails
    audio_tracks: List[AudioTrack] = Field(default_factory=list)
    subtitle_tracks: List[SubtitleTrack] = Field(default_factory=list)

    # Camera and creation metadata
    camera: CameraDetails

    # Generated content
    thumbnails: List[str] = Field(default_factory=list)

    # Analysis results
    analysis: AnalysisDetails

    # Processing metadata
    processing_status: Optional[str] = None
    processing_progress: Optional[int] = None
    current_step: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Search and discovery
    searchable_content: Optional[str] = None
    content_embeddings: Optional[List[float]] = None

# =====================================================
# PIPELINE CONFIGURATION MODELS
# =====================================================

class StepConfiguration(BaseModel):
    """Enhanced step configuration"""
    name: str
    enabled: bool = True
    priority: int = Field(default=50, ge=1, le=100)
    queue: str = "default"
    retry: int = 3
    timeout: int = 300
    params: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)

    # Advanced configuration
    save_immediately: bool = True
    save_partial: bool = False
    requires_auth: bool = True
    output_artifacts: List[str] = Field(default_factory=list)

class PipelineConfiguration(BaseModel):
    """Enhanced pipeline configuration"""
    name: str
    description: str = ""
    version: str = "1.0"

    # Steps configuration
    steps: List[StepConfiguration] = Field(default_factory=list)

    # Global settings
    global_settings: Dict[str, Any] = Field(default_factory=dict)

    # Worker configuration
    worker_config: Dict[str, int] = Field(default_factory=lambda: {
        "default": 4,
        "metadata": 8,
        "thumbnails": 4,
        "compression": 2,
        "ai_analysis": 1,
        "embeddings": 4
    })

    # Processing settings
    enable_incremental_saves: bool = True
    enable_progress_tracking: bool = True
    enable_error_recovery: bool = True

    # Output settings
    create_individual_files: bool = True
    organize_by_timestamp: bool = True
    generate_summary_files: bool = True