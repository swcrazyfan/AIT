from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class ProcessingStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial" # Added as per plan's BaseStep logic for partial saves/progress
    DUPLICATE = "duplicate" # Added based on MD5ChecksumStep logic

class VideoMetadata(BaseModel):
    """Core video metadata"""
    video_id: str = Field(validation_alias='id') # Allow 'id' from DB data to populate 'video_id'
    file_path: str
    file_name: str
    checksum: Optional[str] = None
    file_size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None
    
    # Technical metadata (as per plan, some might be in a nested model later)
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None # Plan uses frame_rate in some places, fps here
    codec: Optional[str] = None
    bitrate: Optional[int] = None # Plan uses bit_rate in some places
    
    # Processing metadata
    status: ProcessingStatus = ProcessingStatus.QUEUED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    processed_steps: List[str] = Field(default_factory=list) # Ensure default is a new list
    
    # Artifacts from processing
    thumbnails: List[str] = Field(default_factory=list) # Ensure default is a new list
    compressed_path: Optional[str] = None
    
    # Analysis results (can be expanded into more detailed models later)
    ai_analysis: Optional[Dict[str, Any]] = None # General dict for now
    transcript: Optional[str] = None # Simple text transcript for now
    embeddings: Optional[List[float]] = None # Placeholder for embeddings vector
    
    # User metadata
    user_id: Optional[str] = None # Important for RLS
    tags: List[str] = Field(default_factory=list) # Ensure default is a new list

    # Fields from MD5ChecksumStep
    is_duplicate: Optional[bool] = None
    duplicate_of: Optional[str] = None
    duplicate_file_name: Optional[str] = None

    # Fields from FFmpegExtractorStep (some overlap, ensure consistency)
    has_audio: Optional[bool] = None
    audio_tracks_count: Optional[int] = None # Renamed from audio_tracks to avoid conflict with detailed audio_tracks list
    format_name: Optional[str] = None

    # Fields from AIFocalLengthStep
    focal_length_ai_category: Optional[str] = None
    focal_length_ai_confidence: Optional[float] = None
    focal_length_source: Optional[str] = None # EXIF, AI, unavailable
    focal_length_mm: Optional[float] = None # From EXIF, if available
    focal_length_category: Optional[str] = None # From EXIF, if available

    # Fields from ExposureAnalysisStep
    exposure_warning: Optional[bool] = None
    exposure_stops: Optional[float] = None
    overexposed_percentage: Optional[float] = None
    underexposed_percentage: Optional[float] = None
    overall_exposure_quality: Optional[str] = None

    # Fields from HDR Extractor (example, assuming they'd update VideoMetadata)
    is_hdr: Optional[bool] = None
    hdr_format: Optional[str] = None # e.g. HDR10, HLG, Dolby Vision
    # ... other HDR fields like color_space, transfer_characteristics etc. can be added
    # or nested if this model becomes too large.

    class Config:
        use_enum_values = True # To store enum values as strings
        populate_by_name = True # Allows use of field names or aliases for population

class StepConfig(BaseModel):
    """Configuration for a processing step"""
    enabled: bool = True
    priority: int = Field(default=50, ge=1, le=100)
    queue: str = "default"
    retry: int = 3 # Max retries for the step
    timeout: int = 300 # Timeout in seconds for the step execution
    concurrency: int = 1 # Concurrency for this specific step type (if supported by worker)
    params: Dict[str, Any] = Field(default_factory=dict) # Step-specific parameters
    continue_on_failure: bool = False # Whether pipeline should continue if this step fails

class StepResult(BaseModel):
    """Result from a step execution"""
    success: bool
    step_name: str
    video_id: str
    
    # Output data that gets merged into video metadata
    data: Dict[str, Any] = Field(default_factory=dict)
    
    # File artifacts created by the step (e.g., {"thumbnail_small": "/path/to/thumb.jpg"})
    artifacts: Dict[str, str] = Field(default_factory=dict)
    
    # Step-specific metadata (e.g., model used, confidence scores not part of main data)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Error information
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None # For more structured error info
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    def model_post_init(self, __context: Any) -> None:
        if self.completed_at is None and (self.success or self.error is not None):
            self.completed_at = datetime.utcnow()
        if self.completed_at and self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()

class PipelineConfig(BaseModel):
    """Complete pipeline configuration"""
    name: str
    description: str = ""
    version: str = "1.0"
    
    # Global settings applicable to all steps or the pipeline itself
    global_settings: Dict[str, Any] = Field(default_factory=dict)
    
    # Steps in order of execution
    # Each dict in the list should match structure that can be parsed into StepConfig
    # plus 'category' and 'name' for step identification.
    # Example: {"category": "metadata", "name": "ffmpeg_extractor", "config": {"enabled": True, ...}}
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Worker configuration (concurrency per queue)
    worker_config: Dict[str, int] = Field(default_factory=lambda: {
        "default": 4,
        "metadata": 8,
        "thumbnails": 4,
        "compression": 2,
        "ai_analysis": 1,
        "embeddings": 4
    })

    # Parallel execution groups (optional, for advanced pipelines)
    # Example: {"group_name": ["step_name1", "step_name2"], ...}
    parallel_groups: Optional[Dict[str, List[str]]] = None