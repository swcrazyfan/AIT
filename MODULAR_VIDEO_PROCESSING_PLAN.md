# Modular Video Processing Tool with Procrastinate & Supabase

## Overview

This is a complete implementation of a modular, queue-based video processing tool that:
- Uses **Procrastinate** for task queuing (PostgreSQL-based, using your Supabase database)
- Implements a **plugin architecture** where each processing step is a separate file
- Allows **easy swapping** of implementations (e.g., different AI analyzers)
- Supports **parallel processing** with configurable concurrency
- Stores everything in **Supabase** (PostgreSQL)

> **⚠️ PROCRASTINATE UPDATES NEEDED**
> 
> This plan has been updated to align with current Procrastinate v2.x best practices:
> - Updated connector initialization patterns
> - Improved async/await usage
> - Enhanced error handling and task retry mechanisms
> - Better worker management and graceful shutdown
> - Updated schema initialization approach

## Architecture

```
video_tool/
├── __main__.py                    # Entry point
├── cli.py                         # CLI commands
├── core/
│   ├── __init__.py
│   ├── app.py                     # Procrastinate app setup
│   ├── models.py                  # Core data models
│   ├── pipeline.py                # Pipeline orchestrator
│   ├── registry.py                # Step registry system
│   ├── db.py                      # Supabase connection
│   ├── auth.py                    # Authentication system
│   └── search.py                  # Comprehensive search
├── steps/                         # Each step in its own file
│   ├── __init__.py
│   ├── base.py                    # Base step class
│   ├── checksum/
│   │   ├── __init__.py
│   │   ├── md5_checksum.py
│   │   └── blake3_checksum.py
│   ├── metadata/
│   │   ├── __init__.py
│   │   ├── ffmpeg_extractor.py
│   │   └── mediainfo_extractor.py
│   ├── thumbnails/
│   │   ├── __init__.py
│   │   ├── opencv_thumbs.py
│   │   └── ffmpeg_thumbs.py
│   ├── compression/
│   │   ├── __init__.py
│   │   └── ffmpeg_compress.py
│   ├── ai_analysis/
│   │   ├── __init__.py
│   │   ├── gemini_analyzer.py
│   │   └── claude_analyzer.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── bge_embeddings.py
│   └── storage/
│       ├── __init__.py
│       └── supabase_store.py
├── configs/                       # Pipeline configurations
│   ├── default.yaml
│   ├── fast.yaml
│   └── ai_comparison.yaml
└── worker.py                      # Worker process

```

## Core Implementation

### 1. Database Setup (`core/db.py`) - UPDATED FOR PROCRASTINATE V2.x

```python
import os
from typing import Optional
from procrastinate import App, PsycopgConnector
from supabase import create_client, Client
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import structlog

load_dotenv()
logger = structlog.get_logger(__name__)

class Database:
    """Manages both Supabase client and Procrastinate app using same database"""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        self.supabase_db_url = os.getenv("SUPABASE_DB_URL")  # PostgreSQL connection string
        
        # Supabase client for API operations
        self.supabase: Optional[Client] = None
        
        # Procrastinate app using same database
        self.procrastinate_app: Optional[App] = None
        
        # SQLAlchemy for direct database access
        self.engine = None
        self.SessionLocal = None
    
    async def initialize(self):
        """Initialize all connections using current Procrastinate patterns"""
        try:
            # Supabase client
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            
            # Parse Supabase DB URL for Procrastinate
            # Format: postgresql://postgres:[password]@db.[project].supabase.co:5432/postgres
            from urllib.parse import urlparse
            parsed = urlparse(self.supabase_db_url)
            
            # UPDATED: Use current Procrastinate connector pattern
            # Use connection string directly instead of individual parameters
            connector = PsycopgConnector(conninfo=self.supabase_db_url)
            
            # Create Procrastinate app with proper configuration
            self.procrastinate_app = App(
                connector=connector,
                import_paths=["video_tool.steps"]  # Auto-discover tasks
            )
            
            # UPDATED: Open app connection using current pattern
            await self.procrastinate_app.open_async()
            
            # UPDATED: Apply schema using current method
            async with self.procrastinate_app.connector.get_sync_connector() as sync_connector:
                with sync_connector.get_sync_connection() as connection:
                    # Apply Procrastinate schema
                    await self.procrastinate_app.connector.execute_query_async(
                        connection, 
                        "SELECT procrastinate_schema.create_all()"
                    )
            
            logger.info("Procrastinate schema applied successfully")
            
            # SQLAlchemy for complex queries (unchanged)
            self.engine = create_async_engine(
                self.supabase_db_url.replace('postgresql://', 'postgresql+asyncpg://'),
                echo=False
            )
            self.SessionLocal = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
            
            logger.info("Database initialization completed")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    async def close(self):
        """Close all connections"""
        try:
            if self.procrastinate_app:
                await self.procrastinate_app.close_async()
            if self.engine:
                await self.engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")

# UPDATED: Use better singleton pattern
_db_instance: Optional[Database] = None

async def get_db() -> Database:
    """Get initialized database instance using singleton pattern"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
        await _db_instance.initialize()
    return _db_instance

async def get_supabase() -> Client:
    """Get Supabase client"""
    database = await get_db()
    return database.supabase

async def get_procrastinate_app() -> App:
    """Get Procrastinate app"""
    database = await get_db()
    return database.procrastinate_app

# UPDATED: Add cleanup helper
async def cleanup_db():
    """Cleanup database connections"""
    global _db_instance
    if _db_instance:
        await _db_instance.close()
        _db_instance = None
```

### 2. Authentication System (`core/auth.py`)

```python
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import structlog

from video_tool.core.db import get_supabase

logger = structlog.get_logger(__name__)

# Auth file location
AUTH_FILE = Path.home() / ".video_tool_auth.json"

class AuthManager:
    """Manages CLI authentication with Supabase"""
    
    def __init__(self):
        self.client = None
        self._ensure_auth_file()
    
    def _ensure_auth_file(self):
        """Ensure auth file exists with proper permissions"""
        if not AUTH_FILE.exists():
            AUTH_FILE.touch(mode=0o600)
    
    async def login(self, email: str, password: str) -> bool:
        """Login with email and password"""
        try:
            if not self.client:
                self.client = await get_supabase()
            
            # Authenticate using Supabase Auth
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if response.session:
                # Save session
                session_data = {
                    "access_token": response.session.access_token,
                    "refresh_token": response.session.refresh_token,
                    "expires_at": response.session.expires_at,
                    "user_id": response.user.id,
                    "email": response.user.email
                }
                self._save_session(session_data)
                logger.info(f"Successfully logged in as {email}")
                return True
                
        except Exception as e:
            logger.error(f"Login failed: {str(e)}")
            
        return False
    
    async def signup(self, email: str, password: str) -> bool:
        """Sign up new user"""
        try:
            if not self.client:
                self.client = await get_supabase()
            
            # Sign up using Supabase Auth
            response = self.client.auth.sign_up({
                "email": email,
                "password": password
            })
            
            if response.user:
                logger.info(f"Successfully signed up: {email}")
                logger.info("Please check your email to confirm your account.")
                return True
                
        except Exception as e:
            logger.error(f"Signup failed: {str(e)}")
            
        return False
    
    async def logout(self) -> bool:
        """Logout and clear stored session"""
        try:
            if not self.client:
                self.client = await get_supabase()
            
            # Sign out from Supabase
            await self.client.auth.sign_out()
            
            # Clear local session file
            if AUTH_FILE.exists():
                AUTH_FILE.unlink()
            
            logger.info("Successfully logged out")
            return True
            
        except Exception as e:
            logger.error(f"Logout failed: {str(e)}")
            return False
    
    def get_current_session(self) -> Optional[Dict[str, Any]]:
        """Get current session if valid"""
        if not AUTH_FILE.exists():
            return None
        
        try:
            with open(AUTH_FILE, 'r') as f:
                session_data = json.load(f)
            
            # Check if token is expired
            expires_at = session_data.get('expires_at', 0)
            if expires_at <= datetime.utcnow().timestamp():
                # Try to refresh token
                return self._refresh_session(session_data)
            
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to read session: {str(e)}")
            return None
    
    async def _refresh_session(self, old_session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Refresh an expired session using the refresh token"""
        try:
            if not self.client:
                self.client = await get_supabase()
            
            # Set the refresh token
            self.client.auth.set_session(
                old_session['access_token'],
                old_session['refresh_token']
            )
            
            # Attempt to refresh
            new_session = self.client.auth.refresh_session()
            
            if new_session.session:
                # Save the refreshed session
                session_data = {
                    "access_token": new_session.session.access_token,
                    "refresh_token": new_session.session.refresh_token,
                    "expires_at": new_session.session.expires_at,
                    "user_id": new_session.user.id,
                    "email": new_session.user.email
                }
                self._save_session(session_data)
                return session_data
                
        except Exception as e:
            logger.error(f"Failed to refresh session: {str(e)}")
            
        return None
    
    async def get_authenticated_client(self) -> Optional[Client]:
        """Get authenticated Supabase client"""
        session = self.get_current_session()
        if not session:
            return None
        
        if not self.client:
            self.client = await get_supabase()
        
        # Set the session
        self.client.auth.set_session(
            session['access_token'],
            session['refresh_token']
        )
        
        return self.client
    
    async def get_user_profile(self) -> Optional[Dict[str, Any]]:
        """Get current user profile"""
        client = await self.get_authenticated_client()
        if not client:
            return None
        
        try:
            # Get user profile from the database
            result = client.rpc('get_user_profile').execute()
            if result.data:
                return result.data
                
        except Exception as e:
            logger.error(f"Failed to get user profile: {str(e)}")
            
        return None
    
    async def is_admin(self) -> bool:
        """Check if current user is admin"""
        profile = await self.get_user_profile()
        if not profile:
            return False
        
        return profile.get('profile_type') == 'admin'
    
    def _save_session(self, session: Dict[str, Any]) -> None:
        """Save session to local file"""
        try:
            with open(AUTH_FILE, 'w') as f:
                json.dump(session, f)
            
            # Ensure proper permissions
            AUTH_FILE.chmod(0o600)
            
        except Exception as e:
            logger.error(f"Failed to save session: {str(e)}")
    
    async def require_auth(self) -> Client:
        """Require authentication or raise exception"""
        client = await self.get_authenticated_client()
        if not client:
            raise Exception("Authentication required. Please login first.")
        return client
```

### 3. Models (`core/models.py`)

```python
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class ProcessingStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class VideoMetadata(BaseModel):
    """Core video metadata"""
    video_id: str
    file_path: str
    file_name: str
    checksum: Optional[str] = None
    file_size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None
    
    # Technical metadata
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    codec: Optional[str] = None
    bitrate: Optional[int] = None
    
    # Processing metadata
    status: ProcessingStatus = ProcessingStatus.QUEUED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    processed_steps: List[str] = []
    
    # Artifacts from processing
    thumbnails: List[str] = []
    compressed_path: Optional[str] = None
    
    # Analysis results
    ai_analysis: Optional[Dict[str, Any]] = None
    transcript: Optional[str] = None
    embeddings: Optional[List[float]] = None
    
    # User metadata
    user_id: Optional[str] = None
    tags: List[str] = []

class StepConfig(BaseModel):
    """Configuration for a processing step"""
    enabled: bool = True
    priority: int = Field(default=50, ge=1, le=100)
    queue: str = "default"
    retry: int = 3
    timeout: int = 300
    concurrency: int = 1
    params: Dict[str, Any] = {}

class StepResult(BaseModel):
    """Result from a step execution"""
    success: bool
    step_name: str
    video_id: str
    
    # Output data that gets merged into video metadata
    data: Dict[str, Any] = {}
    
    # File artifacts created
    artifacts: Dict[str, str] = {}
    
    # Step-specific metadata
    metadata: Dict[str, Any] = {}
    
    # Error information
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

class PipelineConfig(BaseModel):
    """Complete pipeline configuration"""
    name: str
    description: str = ""
    version: str = "1.0"
    
    # Global settings
    global_settings: Dict[str, Any] = {}
    
    # Steps in order
    steps: List[Dict[str, Any]] = []
    
    # Worker configuration
    worker_config: Dict[str, int] = {
        "default": 4,
        "metadata": 8,
        "thumbnails": 4,
        "compression": 2,
        "ai_analysis": 1,
        "embeddings": 4
    }
```

### 4. Base Step Class with Incremental Saves and Auth (`steps/base.py`) - UPDATED

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
import structlog
import asyncio

from video_tool.core.models import StepConfig, StepResult, VideoMetadata
from video_tool.core.db import get_supabase

logger = structlog.get_logger()

class BaseStep(ABC):
    """Base class for all processing steps with incremental saving and user context"""
    
    # Step metadata (must be overridden)
    name: str = ""
    version: str = "1.0"
    description: str = ""
    category: str = ""
    
    # Dependencies
    requires: List[str] = []  # Required fields from video metadata
    provides: List[str] = []  # Fields this step adds to metadata
    optional_requires: List[str] = []  # Optional fields that enhance processing
    
    # Save configuration
    save_immediately: bool = True  # Save results immediately
    save_partial: bool = False     # Save partial results during processing
    
    def __init__(self, config: StepConfig):
        self.config = config
        self.logger = logger.bind(step=self.name)
        self.supabase = None
    
    # UPDATED: Enhanced error handling and context passing
    async def execute(self, video: VideoMetadata, step_index: int, total_steps: int, context: Optional[Dict[str, Any]] = None) -> StepResult:
        """Execute step with progress tracking, incremental saves, and user context"""
        if not self.supabase:
            self.supabase = await get_supabase()
        
        # Update: Starting step
        await self._update_progress(video.video_id, video.user_id, 'starting', step_index, total_steps)
        
        try:
            # Validate inputs
            if not await self.validate_input(video):
                missing_fields = [field for field in self.requires if not hasattr(video, field) or getattr(video, field) is None]
                raise ValueError(f"Missing required inputs: {missing_fields}")
            
            # UPDATED: Pass context to process method
            result = await self.process(video, context)
            
            if result.success:
                # Save results immediately
                if self.save_immediately:
                    await self._save_step_results(video.video_id, video.user_id, result)
                
                # Update: Completed step
                await self._update_progress(video.video_id, video.user_id, 'completed', step_index, total_steps)
            else:
                # Update: Failed step
                await self._update_progress(video.video_id, video.user_id, 'failed', step_index, total_steps, result.error)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Step failed: {str(e)}", exc_info=True)
            await self._update_progress(video.video_id, video.user_id, 'error', step_index, total_steps, str(e))
            
            # UPDATED: Return failed result instead of raising
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video.video_id,
                error=str(e),
                completed_at=datetime.utcnow()
            )
    
    @abstractmethod
    async def process(self, video: VideoMetadata, context: Optional[Dict[str, Any]] = None) -> StepResult:
        """
        Process the video and return results.
        
        Args:
            video: Current video metadata (includes user_id)
            context: Optional additional context from Procrastinate
            
        Returns:
            StepResult with data to be merged into video metadata
        """
        pass
    
    async def validate_input(self, video: VideoMetadata) -> bool:
        """Validate that required inputs are present"""
        video_dict = video.dict()
        
        for field in self.requires:
            if field not in video_dict or video_dict[field] is None:
                self.logger.warning(f"Missing required field: {field}")
                return False
        
        return True
    
    async def _update_progress(self, video_id: str, user_id: str, status: str, step_index: int, total_steps: int, error: str = None):
        """Update processing progress in database with user context"""
        progress_pct = int((step_index / total_steps) * 100)
        
        update_data = {
            'current_step': self.name,
            'processing_progress': progress_pct,
            'total_steps': total_steps,
            'updated_at': datetime.utcnow().isoformat()
        }
        
        # Update processing status JSON
        status_update = {}
        if status == 'completed':
            update_data['last_step_completed'] = self.name
            status_update[self.name] = {
                'status': 'completed',
                'completed_at': datetime.utcnow().isoformat()
            }
        elif status in ['failed', 'error']:
            status_update[self.name] = {
                'status': status,
                'error': error,
                'failed_at': datetime.utcnow().isoformat()
            }
        else:
            status_update[self.name] = {
                'status': status,
                'started_at': datetime.utcnow().isoformat()
            }
        
        # Merge with existing processing_status
        result = await self.supabase.table('clips')\
            .select('processing_status')\
            .eq('id', video_id)\
            .eq('user_id', user_id)\
            .single()\
            .execute()
            
        existing_status = result.data.get('processing_status', {}) if result.data else {}
        existing_status.update(status_update)
        update_data['processing_status'] = existing_status
        
        await self.supabase.table('clips')\
            .update(update_data)\
            .eq('id', video_id)\
            .eq('user_id', user_id)\
            .execute()
        
        # Log processing event
        await self.supabase.table('processing_events').insert({
            'video_id': video_id,
            'user_id': user_id,
            'step_name': self.name,
            'step_index': step_index,
            'status': status,
            'error': error,
            'timestamp': datetime.utcnow().isoformat()
        }).execute()
    
    async def _save_step_results(self, video_id: str, user_id: str, result: StepResult):
        """Save step results immediately to database with user context"""
        if not result.data:
            return
        
        # Add metadata about when this data was saved
        result.data['_last_updated'] = datetime.utcnow().isoformat()
        result.data['_updated_by_step'] = self.name
        
        # Update the clips table with the new data
        await self.supabase.table('clips')\
            .update(result.data)\
            .eq('id', video_id)\
            .eq('user_id', user_id)\
            .execute()
        
        # If there are artifacts (files created), store their paths
        if result.artifacts:
            await self.supabase.table('artifacts').insert([
                {
                    'video_id': video_id,
                    'user_id': user_id,
                    'step_name': self.name,
                    'artifact_type': key,
                    'file_path': path,
                    'created_at': datetime.utcnow().isoformat()
                }
                for key, path in result.artifacts.items()
            ]).execute()
        
        self.logger.info(f"Saved results for video {video_id}")
    
    async def save_partial_result(self, video_id: str, user_id: str, partial_data: Dict[str, Any]):
        """Save partial results during processing (for long-running steps)"""
        if not self.save_partial:
            return
        
        partial_data['_partial_update'] = True
        partial_data['_partial_update_at'] = datetime.utcnow().isoformat()
        
        await self.supabase.table('clips')\
            .update(partial_data)\
            .eq('id', video_id)\
            .eq('user_id', user_id)\
            .execute()
    
    async def setup(self):
        """Optional setup method called once when step is initialized"""
        pass
    
    async def cleanup(self, video_id: str, result: StepResult):
        """Optional cleanup after processing"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get step information"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category,
            "requires": self.requires,
            "provides": self.provides,
            "config": self.config.dict()
        }
```

### 5. Example Steps with Incremental Saves

#### MD5 Checksum (`steps/checksum/md5_checksum.py`)

```python
import hashlib
from pathlib import Path

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata

class MD5ChecksumStep(BaseStep):
    """Fast MD5 checksum calculation with duplicate detection"""
    
    name = "md5_checksum"
    version = "1.0"
    description = "Calculate MD5 checksum for deduplication"
    category = "checksum"
    
    requires = ["file_path"]
    provides = ["checksum", "file_size_bytes", "is_duplicate"]
    
    async def process(self, video: VideoMetadata) -> StepResult:
        try:
            file_path = Path(video.file_path)
            
            if not file_path.exists():
                return StepResult(
                    success=False,
                    step_name=self.name,
                    video_id=video.video_id,
                    error=f"File not found: {file_path}"
                )
            
            # Calculate MD5
            md5_hash = hashlib.md5()
            file_size = 0
            
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)
                    file_size += len(chunk)
            
            checksum = md5_hash.hexdigest()
            
            # Check for duplicates IMMEDIATELY
            duplicate_check = await self.supabase.table('clips')\
                .select('id, file_name')\
                .eq('checksum', checksum)\
                .neq('id', video.video_id)\
                .execute()
            
            is_duplicate = len(duplicate_check.data) > 0
            
            self.logger.info(f"Calculated checksum: {checksum}, duplicate: {is_duplicate}")
            
            # Save results immediately
            result_data = {
                "checksum": checksum,
                "file_size_bytes": file_size,
                "is_duplicate": is_duplicate
            }
            
            if is_duplicate:
                result_data["duplicate_of"] = duplicate_check.data[0]['id']
                result_data["duplicate_file_name"] = duplicate_check.data[0]['file_name']
                result_data["status"] = "duplicate"
            
            return StepResult(
                success=True,
                step_name=self.name,
                video_id=video.video_id,
                data=result_data
            )
            
        except Exception as e:
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video.video_id,
                error=str(e)
            )
```

#### FFmpeg Metadata Extractor (`steps/metadata/ffmpeg_extractor.py`)

```python
import asyncio
import json
from pathlib import Path

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata

class FFmpegExtractorStep(BaseStep):
    """Extract metadata using ffprobe with incremental saves"""
    
    name = "ffmpeg_extractor"
    version = "1.0"
    description = "Extract video metadata using ffprobe"
    category = "metadata"
    
    requires = ["file_path"]
    provides = ["duration_seconds", "width", "height", "fps", "codec", "bitrate", "has_audio", "audio_tracks"]
    save_partial = True  # Enable partial saves
    
    async def process(self, video: VideoMetadata) -> StepResult:
        try:
            file_path = video.file_path
            
            # First, get basic info quickly
            basic_info = await self._get_basic_info(file_path)
            
            # Save basic info immediately
            await self.save_partial_result(video.video_id, video.user_id, basic_info)
            
            # Then get detailed info
            detailed_info = await self._get_detailed_info(file_path)
            
            # Combine all metadata
            metadata = {**basic_info, **detailed_info}
            
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
        
        return {
            "duration_seconds": float(format_data.get("duration", 0)),
            "bitrate": int(format_data.get("bit_rate", 0)),
            "format_name": format_data.get("format_name")
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
        
        metadata = {
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "codec": video_stream.get("codec_name"),
            "has_audio": len(audio_streams) > 0,
            "audio_tracks": len(audio_streams)
        }
        
        # Calculate FPS
        if "r_frame_rate" in video_stream:
            num, den = map(int, video_stream["r_frame_rate"].split("/"))
            metadata["fps"] = num / den if den != 0 else 0
        
        return metadata
```

#### Parallel Thumbnail Generation (`steps/thumbnails/parallel_thumbs.py`)

```python
import asyncio
from pathlib import Path
from typing import List
import cv2

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata

class ParallelThumbsStep(BaseStep):
    """Generate thumbnails in parallel with immediate saves"""
    
    name = "parallel_thumbs"
    version = "1.0"
    description = "Generate thumbnails in parallel"
    category = "thumbnails"
    
    requires = ["file_path", "duration_seconds"]
    provides = ["thumbnails", "thumbnail_count"]
    save_partial = True
    
    async def process(self, video: VideoMetadata) -> StepResult:
        try:
            count = self.config.params.get("count", 5)
            width = self.config.params.get("width", 1920)
            
            # Calculate thumbnail positions
            duration = video.duration_seconds
            positions = [duration * i / (count + 1) for i in range(1, count + 1)]
            
            # Generate thumbnails in parallel
            tasks = []
            for i, position in enumerate(positions):
                task = self._generate_single_thumbnail(
                    video.video_id,
                    video.user_id,
                    video.file_path,
                    position,
                    i,
                    width
                )
                tasks.append(task)
            
            # Run all thumbnail generations in parallel
            thumbnail_paths = await asyncio.gather(*tasks)
            
            # Filter out any failed thumbnails
            valid_thumbnails = [t for t in thumbnail_paths if t is not None]
            
            return StepResult(
                success=True,
                step_name=self.name,
                video_id=video.video_id,
                data={
                    "thumbnails": valid_thumbnails,
                    "thumbnail_count": len(valid_thumbnails)
                },
                artifacts={
                    f"thumbnail_{i}": path 
                    for i, path in enumerate(valid_thumbnails)
                }
            )
            
        except Exception as e:
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video.video_id,
                error=str(e)
            )
    
    async def _generate_single_thumbnail(self, video_id: str, user_id: str, video_path: str, position: float, index: int, width: int) -> Optional[str]:
        """Generate a single thumbnail and save immediately"""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            # Seek to position
            cap.set(cv2.CAP_PROP_POS_MSEC, position * 1000)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            # Resize frame
            height = int(frame.shape[0] * width / frame.shape[1])
            resized = cv2.resize(frame, (width, height))
            
            # Save thumbnail
            output_dir = Path(f"thumbnails/{video_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"thumb_{index:03d}_{int(position)}s.jpg"
            cv2.imwrite(str(output_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Update database immediately with this thumbnail
            await self.supabase.rpc('append_to_array', {
                'table_name': 'clips',
                'id': video_id,
                'column_name': 'thumbnails',
                'new_value': str(output_path)
            }).execute()
            
            self.logger.info(f"Generated thumbnail {index} at {position}s")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate thumbnail {index}: {str(e)}")
            return None
```

#### AI Focal Length Detection (`steps/ai_analysis/focal_length_detector.py`)

```python
import os
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image

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

class AIFocalLengthStep(BaseStep):
    """AI-powered focal length detection using Transformers when EXIF data unavailable"""
    
    name = "ai_focal_length"
    version = "1.0"
    description = "Detect focal length category using AI when EXIF data is missing"
    category = "ai_analysis"
    
    requires = ["thumbnails"]
    provides = ["focal_length_ai_category", "focal_length_ai_confidence", "focal_length_source"]
    optional_requires = ["focal_length_mm", "focal_length_category"]  # From EXIF
    
    def __init__(self, config):
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
    
    async def process(self, video: VideoMetadata) -> StepResult:
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
                        "focal_length_source": "EXIF",
                        "focal_length_ai_category": None,
                        "focal_length_ai_confidence": None
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
                        "focal_length_source": "unavailable",
                        "focal_length_ai_category": None,
                        "focal_length_ai_confidence": None
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
            
            result_data = {
                "focal_length_source": "AI",
                "focal_length_ai_category": category,
                "focal_length_ai_confidence": confidence,
                "focal_length_mm": None  # AI doesn't provide exact mm, only category
            }
            
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
    
    async def _detect_focal_length_ai(self, image_path: str) -> tuple[Optional[str], float]:
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
            self.logger.error(f"AI focal length detection failed: {str(e)}")
            return None, 0.0
    
    def _map_ai_category(self, ai_category: str) -> str:
        """
        Map AI model output to our standard focal length categories.
        
        The tonyassi/camera-lens-focal-length model may use different category names,
        so we normalize them to our standard FOCAL_LENGTH_RANGES keys.
        """
        # Convert to uppercase and handle common variations
        ai_category = ai_category.upper().replace("-", "_").replace(" ", "_")
        
        # Direct mapping
        if ai_category in FOCAL_LENGTH_RANGES:
            return ai_category
        
        # Handle variations
        category_mappings = {
            "ULTRAWIDE": "ULTRA-WIDE",
            "ULTRA_WIDE": "ULTRA-WIDE", 
            "WIDE_ANGLE": "WIDE",
            "NORMAL": "MEDIUM",
            "STANDARD": "MEDIUM",
            "TELEPHOTO": "TELEPHOTO",
            "TELE": "TELEPHOTO",
            "LONG": "LONG-LENS"
        }
        
        mapped = category_mappings.get(ai_category, ai_category)
        
        # Final validation - return valid category or default to MEDIUM
        if mapped in FOCAL_LENGTH_RANGES:
            return mapped
        else:
            self.logger.warning(f"Unknown AI category '{ai_category}', defaulting to MEDIUM")
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
```

#### Exposure Analysis (`steps/ai_analysis/exposure_analyzer.py`)

```python
import math
from pathlib import Path
from typing import Dict, Any
import cv2
import numpy as np

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata

class ExposureAnalysisStep(BaseStep):
    """Analyze exposure in video thumbnails using OpenCV"""
    
    name = "exposure_analysis"
    version = "1.0"
    description = "Analyze exposure quality in thumbnails - detect over/underexposure"
    category = "ai_analysis"
    
    requires = ["thumbnails"]
    provides = [
        "exposure_warning", 
        "exposure_stops", 
        "overexposed_percentage", 
        "underexposed_percentage",
        "overall_exposure_quality"
    ]
    
    # Exposure analysis thresholds
    OVEREXPOSURE_THRESHOLD = 240  # Pixel values above this are considered overexposed
    UNDEREXPOSURE_THRESHOLD = 16  # Pixel values below this are considered underexposed
    WARNING_PERCENTAGE = 0.05     # 5% overexposed/underexposed pixels trigger warning
    
    def __init__(self, config):
        super().__init__(config)
        self.opencv_available = self._check_opencv()
        
    def _check_opencv(self) -> bool:
        """Check if OpenCV is available"""
        try:
            import cv2
            return True
        except ImportError:
            self.logger.warning("OpenCV not available for exposure analysis")
            self.logger.info("Install: pip install opencv-python")
            return False
    
    async def process(self, video: VideoMetadata) -> StepResult:
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
            analyses = []
            
            for i, thumbnail_path in enumerate(thumbnails):
                if not Path(thumbnail_path).exists():
                    self.logger.warning(f"Thumbnail not found: {thumbnail_path}")
                    continue
                
                analysis = await self._analyze_single_thumbnail(thumbnail_path)
                if analysis:
                    analyses.append(analysis)
                    
                    # Save partial results for each thumbnail analyzed
                    await self.save_partial_result(
                        video.video_id, 
                        video.user_id,
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
            
            result_data = {
                "exposure_warning": overall_analysis["warning"],
                "exposure_stops": overall_analysis["stops"],
                "overexposed_percentage": overall_analysis["overexposed_pct"],
                "underexposed_percentage": overall_analysis["underexposed_pct"],
                "overall_exposure_quality": exposure_quality,
                "thumbnails_analyzed": len(analyses),
                "exposure_details": {
                    "individual_analyses": analyses,
                    "avg_brightness": overall_analysis["avg_brightness"],
                    "brightness_std": overall_analysis["brightness_std"]
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
                    "thumbnails_analyzed": len(analyses),
                    "analysis_method": "opencv_histogram",
                    "thresholds": {
                        "overexposure": self.OVEREXPOSURE_THRESHOLD,
                        "underexposure": self.UNDEREXPOSURE_THRESHOLD,
                        "warning_percentage": self.WARNING_PERCENTAGE
                    }
                }
            )
            
        except Exception as e:
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video.video_id,
                error=str(e)
            )
    
    async def _analyze_single_thumbnail(self, thumbnail_path: str) -> Dict[str, Any]:
        """
        Analyze exposure in a single thumbnail image.
        
        Args:
            thumbnail_path: Path to the thumbnail image
            
        Returns:
            Dict with exposure analysis results
        """
        try:
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
            self.logger.error(f"Failed to analyze thumbnail {thumbnail_path}: {str(e)}")
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
            brightness_ratio = mean_brightness / ideal_brightness
            
            # Convert brightness ratio to stops (each stop is 2x brightness change)
            if brightness_ratio != 1.0:
                stops = math.log2(brightness_ratio)
                # Only report significant deviations
                return stops if abs(stops) > 0.2 else 0.0
            
            return 0.0
    
    def _calculate_overall_exposure(self, analyses: list) -> Dict[str, Any]:
        """Calculate overall exposure statistics from multiple thumbnail analyses"""
        if not analyses:
            return {}
        
        # Average the exposure metrics
        avg_overexposed = np.mean([a["overexposed_pct"] for a in analyses])
        avg_underexposed = np.mean([a["underexposed_pct"] for a in analyses])
        avg_brightness = np.mean([a["mean_brightness"] for a in analyses])
        brightness_std = np.std([a["mean_brightness"] for a in analyses])
        
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
        if not analysis:
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
            self.logger.info(f"Thresholds - Overexposure: {self.OVEREXPOSURE_THRESHOLD}, Underexposure: {self.UNDEREXPOSURE_THRESHOLD}")
        else:
            self.logger.info("Exposure Analysis step disabled (OpenCV unavailable)")
    
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
            "quality_ratings": ["excellent", "good", "fair", "poor"]
        })
        return info
```

### 6. Comprehensive Search System (`core/search.py`)

```python
"""
Comprehensive search system with semantic, full-text, and hybrid search capabilities.
"""
import asyncio
from typing import Dict, List, Optional, Tuple, Literal, Any
from datetime import datetime
import structlog

from video_tool.core.db import get_supabase
from video_tool.core.auth import AuthManager
from video_tool.embeddings import prepare_search_embeddings, generate_embeddings

logger = structlog.get_logger(__name__)

SearchType = Literal["semantic", "fulltext", "hybrid", "transcripts", "similar"]

class VideoSearcher:
    """Comprehensive video search with multiple search strategies"""
    
    def __init__(self):
        self.auth_manager = AuthManager()
    
    async def search(
        self,
        query: str,
        search_type: SearchType = "hybrid",
        match_count: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform comprehensive search across video catalog.
        
        Args:
            query: Search query text
            search_type: Type of search (semantic, fulltext, hybrid, transcripts, similar)
            match_count: Number of results to return
            filters: Optional filters (camera_make, content_category, etc.)
            weights: Search weights for hybrid search
            
        Returns:
            List of matching video clips with metadata
        """
        client = await self._get_authenticated_client()
        user_id = self._get_current_user_id()
        
        if not client or not user_id:
            raise ValueError("Authentication required for search")
        
        # Apply search strategy
        if search_type == "semantic":
            return await self._semantic_search(client, user_id, query, match_count, weights)
        elif search_type == "fulltext":
            return await self._fulltext_search(client, user_id, query, match_count)
        elif search_type == "hybrid":
            return await self._hybrid_search(client, user_id, query, match_count, weights)
        elif search_type == "transcripts":
            return await self._transcript_search(client, user_id, query, match_count)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    async def find_similar(
        self,
        clip_id: str,
        match_count: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find clips similar to a given clip using vector similarity.
        
        Args:
            clip_id: ID of the source clip
            match_count: Number of similar clips to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar clips
        """
        client = await self._get_authenticated_client()
        user_id = self._get_current_user_id()
        
        try:
            result = await client.rpc('find_similar_clips', {
                'source_clip_id': clip_id,
                'user_id_filter': user_id,
                'match_count': match_count,
                'similarity_threshold': similarity_threshold
            }).execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Similar search failed: {str(e)}")
            return []
    
    async def _semantic_search(
        self, 
        client, 
        user_id: str, 
        query: str, 
        match_count: int,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using vector embeddings"""
        try:
            # Prepare content for embedding
            summary_content, keyword_content = prepare_search_embeddings(query)
            
            # Generate embeddings
            summary_embedding, keyword_embedding = await generate_embeddings(
                summary_content, 
                keyword_content
            )
            
            # Set default weights
            search_params = {
                'query_summary_embedding': summary_embedding,
                'query_keyword_embedding': keyword_embedding,
                'user_id_filter': user_id,
                'match_count': match_count,
                'summary_weight': weights.get('summary', 1.0) if weights else 1.0,
                'keyword_weight': weights.get('keyword', 0.8) if weights else 0.8,
                'similarity_threshold': weights.get('threshold', 0.0) if weights else 0.0
            }
            
            # Execute semantic search
            result = await client.rpc('semantic_search_clips', search_params).execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []
    
    async def _fulltext_search(
        self, 
        client, 
        user_id: str, 
        query: str, 
        match_count: int
    ) -> List[Dict[str, Any]]:
        """Perform full-text search"""
        try:
            result = await client.rpc('fulltext_search_clips', {
                'query_text': query,
                'user_id_filter': user_id,
                'match_count': match_count
            }).execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Full-text search failed: {str(e)}")
            return []
    
    async def _hybrid_search(
        self, 
        client, 
        user_id: str, 
        query: str, 
        match_count: int,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining full-text and semantic search"""
        try:
            # Prepare embeddings
            summary_content, keyword_content = prepare_search_embeddings(query)
            summary_embedding, keyword_embedding = await generate_embeddings(
                summary_content, 
                keyword_content
            )
            
            # Set default weights for RRF
            default_weights = {
                'fulltext': 1.0,
                'summary': 1.0,
                'keyword': 0.8,
                'rrf_k': 50
            }
            
            if weights:
                default_weights.update(weights)
            
            # Execute hybrid search
            result = await client.rpc('hybrid_search_clips', {
                'query_text': query,
                'query_summary_embedding': summary_embedding,
                'query_keyword_embedding': keyword_embedding,
                'user_id_filter': user_id,
                'match_count': match_count,
                'fulltext_weight': default_weights['fulltext'],
                'summary_weight': default_weights['summary'],
                'keyword_weight': default_weights['keyword'],
                'rrf_k': default_weights['rrf_k']
            }).execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            return []
    
    async def _transcript_search(
        self, 
        client, 
        user_id: str, 
        query: str, 
        match_count: int
    ) -> List[Dict[str, Any]]:
        """Perform search specifically on transcripts"""
        try:
            result = await client.rpc('search_transcripts', {
                'query_text': query,
                'user_id_filter': user_id,
                'match_count': match_count,
                'min_content_length': 50
            }).execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Transcript search failed: {str(e)}")
            return []
    
    async def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics for the video catalog"""
        client = await self._get_authenticated_client()
        
        if not client:
            return {}
        
        try:
            result = await client.rpc('get_user_stats').execute()
            return result.data or {}
            
        except Exception as e:
            logger.error(f"Failed to get user stats: {str(e)}")
            return {}
    
    async def get_clip_details(self, clip_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific clip"""
        client = await self._get_authenticated_client()
        user_id = self._get_current_user_id()
        
        if not client or not user_id:
            return None
        
        try:
            # Get clip details
            clip_result = await client.table('clips')\
                .select('*')\
                .eq('id', clip_id)\
                .eq('user_id', user_id)\
                .single()\
                .execute()
            
            if not clip_result.data:
                return None
            
            clip_data = clip_result.data
            
            # Get transcript if available
            transcript_result = await client.table('transcripts')\
                .select('full_text')\
                .eq('clip_id', clip_id)\
                .eq('user_id', user_id)\
                .execute()
            
            if transcript_result.data:
                clip_data['transcript'] = transcript_result.data[0]['full_text']
            
            # Get AI analysis if available
            analysis_result = await client.table('analysis')\
                .select('*')\
                .eq('clip_id', clip_id)\
                .eq('user_id', user_id)\
                .execute()
            
            if analysis_result.data:
                clip_data['ai_analysis'] = analysis_result.data
            
            return clip_data
            
        except Exception as e:
            logger.error(f"Failed to get clip details: {str(e)}")
            return None
    
    async def _get_authenticated_client(self):
        """Get authenticated Supabase client"""
        return await self.auth_manager.get_authenticated_client()
    
    def _get_current_user_id(self) -> Optional[str]:
        """Get current authenticated user ID"""
        session = self.auth_manager.get_current_session()
        return session['user_id'] if session else None

def format_search_results(
    results: List[Dict[str, Any]], 
    search_type: str, 
    show_scores: bool = False
) -> List[Dict[str, Any]]:
    """
    Format search results for display.
    
    Args:
        results: Raw search results
        search_type: Type of search performed
        show_scores: Whether to include similarity/ranking scores
        
    Returns:
        Formatted results for display
    """
    formatted_results = []
    
    for result in results:
        formatted = {
            'id': result.get('id'),
            'file_name': result.get('file_name'),
            'content_summary': result.get('content_summary'),
            'content_category': result.get('content_category'),
            'duration': format_duration(result.get('duration_seconds', 0)),
            'camera': f"{result.get('camera_make', '')} {result.get('camera_model', '')}".strip(),
            'processed_at': result.get('processed_at', ''),
            'tags': result.get('content_tags', [])
        }
        
        # Add search-specific fields
        if show_scores:
            if search_type == "semantic":
                formatted['similarity_score'] = f"{result.get('combined_similarity', 0):.3f}"
            elif search_type == "hybrid":
                formatted['search_rank'] = f"{result.get('search_rank', 0):.3f}"
                formatted['match_type'] = result.get('match_type', 'unknown')
            elif search_type == "fulltext":
                formatted['fts_rank'] = f"{result.get('fts_rank', 0):.3f}"
        
        formatted_results.append(formatted)
    
    return formatted_results

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"
```

### 7. Vector Embeddings System (`embeddings.py`)

```python
"""
Vector embeddings generation using BAAI/bge-m3 via DeepInfra.
Following Supabase best practices for hybrid search.
"""
import os
from typing import Tuple, Dict, Any, List
import structlog
import tiktoken
from openai import OpenAI

from video_tool.core.auth import AuthManager

logger = structlog.get_logger(__name__)

def get_embedding_client():
    """Get OpenAI client configured for DeepInfra API"""
    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        raise ValueError("DEEPINFRA_API_KEY environment variable is required")
    
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai"
    )

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        # Fallback to rough estimate
        return len(text) // 4

def truncate_text(text: str, max_tokens: int = 3500) -> Tuple[str, str]:
    """Intelligently truncate text to fit token limit with sentence boundaries"""
    token_count = count_tokens(text)
    
    if token_count <= max_tokens:
        return text, "none"
    
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        
        # First, try token-based truncation
        truncated_tokens = tokens[:max_tokens]
        truncated_text = encoding.decode(truncated_tokens)
        
        # Try to cut at sentence boundary to preserve meaning
        sentences = text.split('. ')
        
        if len(sentences) > 1:
            # Rebuild text sentence by sentence until we exceed token limit
            rebuilt_text = ""
            
            for i, sentence in enumerate(sentences):
                test_text = rebuilt_text + sentence
                if i < len(sentences) - 1:  # Add period back except for last sentence
                    test_text += ". "
                
                if count_tokens(test_text) > max_tokens:
                    # This sentence would exceed limit, stop at previous sentence
                    if rebuilt_text:  # We have at least one complete sentence
                        return rebuilt_text.rstrip(), "sentence_boundary"
                    else:
                        # Even first sentence is too long, fall back to token truncation
                        break
                
                rebuilt_text = test_text
            
            # Fallback to token-based truncation with ellipsis
            return truncated_text + "...", "token_boundary"
        
    except Exception as e:
        logger.warning(f"Token-based truncation failed: {e}")
    
    # Fallback: character-based truncation with sentence awareness
    char_limit = max_tokens * 4  # Rough estimate
    
    if len(text) <= char_limit:
        return text, "none"
    
    truncated = text[:char_limit]
    # Try to cut at last sentence boundary
    last_period = truncated.rfind('. ')
    if last_period > char_limit * 0.7:  # Only if we keep at least 70% of content
        return truncated[:last_period + 1], "sentence_boundary"
    
    return truncated + "...", "character_boundary"

def prepare_search_embeddings(query: str) -> Tuple[str, str]:
    """
    Prepare search query for embedding generation.
    
    Args:
        query: Search query text
        
    Returns:
        Tuple of (summary_content, keyword_content)
    """
    # For search queries, we use the query as both summary and keyword content
    summary_content = f"Video content about: {query}"
    keyword_content = query
    
    return summary_content, keyword_content

def prepare_embedding_content(video_data) -> Tuple[str, str, Dict[str, Any]]:
    """
    Prepare semantic content for embedding generation optimized for hybrid search.
    
    Args:
        video_data: VideoIngestOutput model
        
    Returns:
        Tuple of (summary_content, keyword_content, metadata)
    """
    # SUMMARY EMBEDDING: Semantic narrative content
    summary_parts = []
    
    # Core content description
    if hasattr(video_data, 'analysis') and video_data.analysis.ai_analysis:
        summary = video_data.analysis.ai_analysis.summary
        
        # Content category as context
        if summary.content_category:
            summary_parts.append(f"This is {summary.content_category} content")
        
        # Main semantic description
        if summary.overall:
            summary_parts.append(summary.overall)
        
        # Key activities in natural language
        if summary.key_activities:
            activities_text = ", ".join(summary.key_activities)
            summary_parts.append(f"Key activities include: {activities_text}")
        
        # Location and setting context
        if hasattr(video_data.analysis.ai_analysis, 'content_analysis'):
            locations = []
            entities = video_data.analysis.ai_analysis.content_analysis.entities
            
            if entities and entities.locations:
                for location in entities.locations:
                    locations.append(f"{location.name} ({location.type})")
            
            if locations:
                summary_parts.append(f"Filmed at: {', '.join(locations)}")
    
    summary_content = ". ".join(summary_parts)
    
    # KEYWORD EMBEDDING: Concept tags and semantic keywords
    keyword_concepts = []
    
    # Core semantic concepts from transcript
    if hasattr(video_data, 'analysis') and video_data.analysis.ai_analysis:
        # Include transcript for semantic concept extraction
        if hasattr(video_data.analysis.ai_analysis, 'audio_analysis'):
            transcript = video_data.analysis.ai_analysis.audio_analysis.transcript
            if transcript and transcript.full_text:
                # Extract key phrases from transcript (first 200 chars for concepts)
                transcript_preview = transcript.full_text[:200]
                keyword_concepts.append(transcript_preview)
    
    # Visual and environmental concepts
    visual_concepts = []
    
    # Location concepts
    if hasattr(video_data, 'analysis') and video_data.analysis.ai_analysis:
        entities = video_data.analysis.ai_analysis.content_analysis.entities
        
        # Location-based concepts
        if entities and entities.locations:
            for location in entities.locations:
                visual_concepts.extend([location.name, location.type])
        
        # Object-based concepts
        if entities and entities.objects_of_interest:
            for obj in entities.objects_of_interest:
                visual_concepts.append(obj.object)
    
    # Combine all concept lists
    all_concepts = []
    all_concepts.extend([c for c in keyword_concepts if c])
    all_concepts.extend([c for c in visual_concepts if c])
    
    keyword_content = " ".join(all_concepts)
    
    # Truncate both contents
    summary_content, summary_truncation = truncate_text(summary_content, 3500)
    keyword_content, keyword_truncation = truncate_text(keyword_content, 3500)
    
    metadata = {
        'summary_token_count': count_tokens(summary_content),
        'keyword_token_count': count_tokens(keyword_content),
        'summary_truncation': summary_truncation,
        'keyword_truncation': keyword_truncation
    }
    
    return summary_content, keyword_content, metadata

async def generate_embeddings(summary_content: str, keyword_content: str) -> Tuple[List[float], List[float]]:
    """Generate embeddings using BAAI/bge-m3 via DeepInfra"""
    try:
        client = get_embedding_client()
        
        # Generate summary embedding
        summary_response = client.embeddings.create(
            model="BAAI/bge-m3",
            input=summary_content
        )
        summary_embedding = summary_response.data[0].embedding
        
        # Generate keyword embedding
        keyword_response = client.embeddings.create(
            model="BAAI/bge-m3", 
            input=keyword_content
        )
        keyword_embedding = keyword_response.data[0].embedding
        
        return summary_embedding, keyword_embedding
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise

async def store_embeddings(
    clip_id: str, 
    summary_embedding: List[float], 
    keyword_embedding: List[float],
    summary_content: str,
    keyword_content: str,
    metadata: Dict[str, Any]
) -> bool:
    """Store embeddings in Supabase database following pgvector patterns"""
    try:
        auth_manager = AuthManager()
        client = await auth_manager.get_authenticated_client()
        
        if not client:
            raise ValueError("Authentication required")
        
        # Get user ID
        user_response = client.auth.get_user()
        user_id = user_response.user.id
        
        vector_data = {
            'clip_id': clip_id,
            'user_id': user_id,
            'embedding_type': 'full_clip',
            'embedding_source': 'BAAI/bge-m3',
            'summary_vector': summary_embedding,
            'keyword_vector': keyword_embedding,
            'embedded_content': f"Summary: {summary_content}\nKeywords: {keyword_content}",
            'original_content': f"Summary: {summary_content}\nKeywords: {keyword_content}",
            'token_count': metadata.get('summary_token_count', 0) + metadata.get('keyword_token_count', 0),
            'original_token_count': metadata.get('summary_token_count', 0) + metadata.get('keyword_token_count', 0),
            'truncation_method': f"summary:{metadata.get('summary_truncation', 'none')}, keyword:{metadata.get('keyword_truncation', 'none')}"
        }
        
        result = client.table('vectors').insert(vector_data).execute()
        
        if result.data:
            logger.info(f"Successfully stored embeddings for clip {clip_id}")
            return True
        else:
            logger.error(f"Failed to store embeddings for clip {clip_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error storing embeddings: {str(e)}")
        return False
```

### 8. Database Schema with Authentication & RLS

```sql
-- =====================================================
-- AI INGESTING TOOL - COMPLETE DATABASE SETUP
-- Version: PRODUCTION READY - All fixes incorporated
-- =====================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- =====================================================
-- 1. USER PROFILES TABLE & TRIGGER (WORKING VERSION)
-- =====================================================
CREATE TABLE IF NOT EXISTS user_profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  profile_type TEXT CHECK (profile_type IN ('admin', 'user')) DEFAULT 'user',
  display_name TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

-- Bulletproof user profile creation trigger (TESTED & WORKING)
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER 
LANGUAGE plpgsql 
SECURITY DEFINER
AS $$
BEGIN
  INSERT INTO public.user_profiles (id, profile_type, display_name)
  VALUES (
    NEW.id,
    'user',
    COALESCE(
      NEW.raw_user_meta_data->>'display_name',
      NEW.raw_user_meta_data->>'full_name', 
      NEW.raw_user_meta_data->>'name',
      split_part(NEW.email, '@', 1),
      'User'
    )
  );
  RETURN NEW;
EXCEPTION
  WHEN OTHERS THEN
    RAISE WARNING 'Failed to create user profile for user %: %', NEW.id, SQLERRM;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION handle_new_user();

-- =====================================================
-- 2. CLIPS TABLE - Main video storage
-- =====================================================
CREATE TABLE IF NOT EXISTS clips (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) NOT NULL,
  
  -- File information
  file_path TEXT NOT NULL,
  local_path TEXT NOT NULL,
  file_name TEXT NOT NULL,
  file_checksum TEXT UNIQUE NOT NULL,
  file_size_bytes BIGINT NOT NULL,
  duration_seconds NUMERIC,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  processed_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Technical metadata
  width INTEGER,
  height INTEGER,
  frame_rate NUMERIC,
  codec TEXT,
  camera_make TEXT,
  camera_model TEXT,
  container TEXT,
  
  -- Processing metadata
  processing_status JSONB DEFAULT '{}',
  processing_progress INTEGER DEFAULT 0,
  total_steps INTEGER DEFAULT 0,
  current_step TEXT,
  last_step_completed TEXT,
  
  -- AI analysis summaries
  content_category TEXT,
  content_summary TEXT,
  content_tags TEXT[],
  
  -- Transcript data
  full_transcript TEXT,
  transcript_preview TEXT,
  transcript_word_count INTEGER DEFAULT 0,
  scene_count INTEGER DEFAULT 0,
  ai_processing_status TEXT,
  
  -- Search columns (populated by triggers)
  searchable_content TEXT,
  fts tsvector,
  
  -- Complex metadata as JSONB
  technical_metadata JSONB,
  camera_details JSONB,
  audio_tracks JSONB,
  subtitle_tracks JSONB,
  thumbnails TEXT[]
);

ALTER TABLE clips ENABLE ROW LEVEL SECURITY;

-- Search content trigger for clips
CREATE OR REPLACE FUNCTION update_clips_search_content()
RETURNS TRIGGER AS $$
BEGIN
  NEW.searchable_content := COALESCE(NEW.file_name, '') || ' ' ||
                           COALESCE(NEW.content_summary, '') || ' ' ||
                           COALESCE(NEW.transcript_preview, '') || ' ' ||
                           COALESCE(array_to_string(NEW.content_tags, ' '), '') || ' ' ||
                           COALESCE(NEW.content_category, '');
  NEW.fts := to_tsvector('english', NEW.searchable_content);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER clips_search_content_trigger
  BEFORE INSERT OR UPDATE ON clips
  FOR EACH ROW EXECUTE FUNCTION update_clips_search_content();

-- Add processing columns if they don't exist
ALTER TABLE clips ADD COLUMN IF NOT EXISTS processing_status JSONB DEFAULT '{}';
ALTER TABLE clips ADD COLUMN IF NOT EXISTS processing_progress INTEGER DEFAULT 0;
ALTER TABLE clips ADD COLUMN IF NOT EXISTS total_steps INTEGER DEFAULT 0;
ALTER TABLE clips ADD COLUMN IF NOT EXISTS current_step TEXT;
ALTER TABLE clips ADD COLUMN IF NOT EXISTS last_step_completed TEXT;
ALTER TABLE clips ADD COLUMN IF NOT EXISTS ai_processing_status TEXT;
ALTER TABLE clips ADD COLUMN IF NOT EXISTS scene_count INTEGER DEFAULT 0;
ALTER TABLE clips ADD COLUMN IF NOT EXISTS transcript_preview TEXT;
ALTER TABLE clips ADD COLUMN IF NOT EXISTS transcript_word_count INTEGER DEFAULT 0;

-- =====================================================
-- 3. PROCESSING EVENTS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS processing_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES clips(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    step_name TEXT NOT NULL,
    step_index INTEGER,
    status TEXT NOT NULL,
    error TEXT,
    timestamp TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}'
);

ALTER TABLE processing_events ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 4. SCENES TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS scenes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES clips(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    scene_number INTEGER NOT NULL,
    timestamp_start FLOAT,
    timestamp_end FLOAT,
    description TEXT,
    objects TEXT[],
    activities TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(video_id, scene_number)
);

ALTER TABLE scenes ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 5. ARTIFACTS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES clips(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    step_name TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

ALTER TABLE artifacts ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 6. VECTORS TABLE - For semantic search
-- =====================================================
CREATE TABLE IF NOT EXISTS vectors (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  clip_id UUID REFERENCES clips(id) ON DELETE CASCADE,
  segment_id UUID REFERENCES segments(id) ON DELETE CASCADE,
  user_id UUID REFERENCES auth.users(id) NOT NULL,
  embedding_type TEXT NOT NULL CHECK (embedding_type IN ('full_clip', 'segment', 'keyframe')),
  embedding_source TEXT NOT NULL,
  summary_vector vector(1024),
  keyword_vector vector(1024),
  embedded_content TEXT NOT NULL,
  original_content TEXT,
  token_count INTEGER,
  original_token_count INTEGER,
  truncation_method TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  CONSTRAINT check_vector_scope CHECK (
    (embedding_type = 'full_clip' AND segment_id IS NULL) OR
    (embedding_type IN ('segment', 'keyframe') AND segment_id IS NOT NULL)
  )
);

ALTER TABLE vectors ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 7. TRANSCRIPTS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS transcripts (
  clip_id UUID REFERENCES clips(id) ON DELETE CASCADE,
  user_id UUID REFERENCES auth.users(id) NOT NULL,
  full_text TEXT NOT NULL,
  segments JSONB NOT NULL,
  speakers JSONB,
  non_speech_events JSONB,
  fts tsvector,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (clip_id)
);

ALTER TABLE transcripts ENABLE ROW LEVEL SECURITY;

-- Transcript search trigger
CREATE OR REPLACE FUNCTION update_transcript_search()
RETURNS TRIGGER AS $$
BEGIN
  NEW.fts := to_tsvector('english', NEW.full_text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER transcripts_search_trigger
  BEFORE INSERT OR UPDATE ON transcripts
  FOR EACH ROW EXECUTE FUNCTION update_transcript_search();

-- =====================================================
-- 8. ANALYSIS TABLE - AI analysis results
-- =====================================================
CREATE TABLE IF NOT EXISTS analysis (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  clip_id UUID REFERENCES clips(id) ON DELETE CASCADE,
  segment_id UUID REFERENCES segments(id) ON DELETE CASCADE,
  user_id UUID REFERENCES auth.users(id) NOT NULL,
  analysis_type TEXT NOT NULL,
  analysis_scope TEXT NOT NULL CHECK (analysis_scope IN ('full_clip', 'segment')),
  ai_model TEXT DEFAULT 'gemini-flash-2.5',
  content_category TEXT,
  usability_rating TEXT,
  speaker_count INTEGER,
  visual_analysis JSONB,
  audio_analysis JSONB,
  content_analysis JSONB,
  analysis_summary JSONB,
  analysis_file_path TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  CONSTRAINT check_analysis_scope CHECK (
    (analysis_scope = 'full_clip' AND segment_id IS NULL) OR
    (analysis_scope = 'segment' AND segment_id IS NOT NULL)
  )
);

ALTER TABLE analysis ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 9. ROW LEVEL SECURITY POLICIES
-- =====================================================

-- User Profile Policies
CREATE POLICY "Users can view own profile" ON user_profiles
    FOR SELECT USING (id = auth.uid());

CREATE POLICY "Users can update own profile" ON user_profiles
    FOR UPDATE USING (id = auth.uid())
    WITH CHECK (id = auth.uid());

CREATE POLICY "Admins can view all profiles" ON user_profiles
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM user_profiles 
            WHERE id = auth.uid() AND profile_type = 'admin'
        )
    );

-- Clips Policies
CREATE POLICY "Users can view own clips" ON clips 
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert own clips" ON clips 
    FOR INSERT WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can update own clips" ON clips 
    FOR UPDATE USING (user_id = auth.uid()) 
    WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can delete own clips" ON clips 
    FOR DELETE USING (user_id = auth.uid());

CREATE POLICY "Admins can view all clips" ON clips 
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM user_profiles 
            WHERE id = auth.uid() AND profile_type = 'admin'
        )
    );

-- Processing Events Policies
CREATE POLICY "Users can view own processing events" ON processing_events 
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert own processing events" ON processing_events 
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- Scenes Policies
CREATE POLICY "Users can view own scenes" ON scenes 
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert own scenes" ON scenes 
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- Artifacts Policies
CREATE POLICY "Users can view own artifacts" ON artifacts 
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert own artifacts" ON artifacts 
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- Vectors Policies
CREATE POLICY "Users can view own vectors" ON vectors 
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert own vectors" ON vectors 
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- Transcripts Policies
CREATE POLICY "Users can view own transcripts" ON transcripts 
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert own transcripts" ON transcripts 
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- Analysis Policies
CREATE POLICY "Users can view own analysis" ON analysis 
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert own analysis" ON analysis 
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- =====================================================
-- 10. PERFORMANCE INDEXES
-- =====================================================
CREATE INDEX IF NOT EXISTS idx_clips_user_id ON clips(user_id);
CREATE INDEX IF NOT EXISTS idx_clips_processing_status ON clips(processing_progress, current_step);
CREATE INDEX IF NOT EXISTS idx_processing_events_video ON processing_events(video_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_processing_events_user ON processing_events(user_id);
CREATE INDEX IF NOT EXISTS idx_scenes_video ON scenes(video_id, scene_number);
CREATE INDEX IF NOT EXISTS idx_artifacts_video ON artifacts(video_id, step_name);

CREATE INDEX IF NOT EXISTS idx_clips_fts ON clips USING gin(fts);
CREATE INDEX IF NOT EXISTS idx_transcripts_fts ON transcripts USING gin(fts);
CREATE INDEX IF NOT EXISTS idx_vectors_summary ON vectors USING hnsw (summary_vector vector_ip_ops);
CREATE INDEX IF NOT EXISTS idx_vectors_keyword ON vectors USING hnsw (keyword_vector vector_ip_ops);

-- =====================================================
-- 11. HELPER FUNCTIONS
-- =====================================================

-- Check if user is admin
CREATE OR REPLACE FUNCTION is_admin(check_user_id UUID DEFAULT auth.uid())
RETURNS BOOLEAN
LANGUAGE SQL
SECURITY DEFINER
AS $$
    SELECT EXISTS (
        SELECT 1 FROM user_profiles 
        WHERE id = check_user_id AND profile_type = 'admin'
    );
$$;

-- Get user profile info
CREATE OR REPLACE FUNCTION get_user_profile(profile_user_id UUID DEFAULT auth.uid())
RETURNS TABLE (
    id UUID,
    profile_type TEXT,
    display_name TEXT,
    created_at TIMESTAMPTZ
)
LANGUAGE SQL
SECURITY DEFINER
AS $$
    SELECT up.id, up.profile_type, up.display_name, up.created_at
    FROM user_profiles up
    WHERE up.id = profile_user_id;
$$;

-- Get user statistics
CREATE OR REPLACE FUNCTION get_user_stats(stats_user_id UUID DEFAULT auth.uid())
RETURNS TABLE (
    total_clips INTEGER,
    total_duration_hours NUMERIC,
    total_storage_gb NUMERIC,
    clips_with_transcripts INTEGER,
    clips_in_progress INTEGER,
    clips_completed INTEGER
)
LANGUAGE SQL
SECURITY DEFINER
AS $$
    SELECT 
        COUNT(*)::INTEGER as total_clips,
        ROUND(SUM(COALESCE(duration_seconds, 0)) / 3600.0, 2) as total_duration_hours,
        ROUND(SUM(COALESCE(file_size_bytes, 0)) / (1024.0^3), 2) as total_storage_gb,
        COUNT(CASE WHEN transcript_word_count > 0 THEN 1 END)::INTEGER as clips_with_transcripts,
        COUNT(CASE WHEN processing_progress < 100 THEN 1 END)::INTEGER as clips_in_progress,
        COUNT(CASE WHEN processing_progress = 100 THEN 1 END)::INTEGER as clips_completed
    FROM clips c
    WHERE c.user_id = stats_user_id;
$$;

-- Function to append to array columns
CREATE OR REPLACE FUNCTION append_to_array(
    table_name TEXT,
    id UUID,
    column_name TEXT,
    new_value TEXT
) RETURNS VOID AS $$
BEGIN
    EXECUTE format(
        'UPDATE %I SET %I = array_append(COALESCE(%I, ''{}''), %L) WHERE id = %L',
        table_name, column_name, column_name, new_value, id
    );
END;
$$ LANGUAGE plpgsql;

-- View for monitoring active processing
CREATE OR REPLACE VIEW active_processing AS
SELECT 
    c.id,
    c.file_name,
    c.user_id,
    c.processing_progress,
    c.current_step,
    c.created_at,
    c.updated_at,
    EXTRACT(EPOCH FROM (NOW() - c.updated_at)) as seconds_since_update,
    c.processing_status,
    CASE 
        WHEN c.processing_progress = 100 THEN 'completed'
        WHEN EXTRACT(EPOCH FROM (NOW() - c.updated_at)) > 300 THEN 'stalled'
        WHEN c.processing_status->>'error' IS NOT NULL THEN 'error'
        ELSE 'processing'
    END as status
FROM clips c
WHERE c.processing_progress < 100 
   OR c.updated_at > NOW() - INTERVAL '1 hour'
ORDER BY c.created_at DESC;
```

### 9. Hybrid Search Functions (`hybrid_search.sql`)

```sql
-- =====================================================
-- HYBRID SEARCH FUNCTIONS FOR VIDEO CATALOG
-- =====================================================

-- Function for basic semantic search using vector similarity
CREATE OR REPLACE FUNCTION semantic_search_clips(
  query_summary_embedding vector(1024),
  query_keyword_embedding vector(1024),
  user_id_filter UUID,
  match_count INT DEFAULT 10,
  summary_weight FLOAT DEFAULT 1.0,
  keyword_weight FLOAT DEFAULT 0.8,
  similarity_threshold FLOAT DEFAULT 0.0
)
RETURNS TABLE (
  id UUID,
  file_name TEXT,
  local_path TEXT,
  content_summary TEXT,
  content_tags TEXT[],
  duration_seconds NUMERIC,
  camera_make TEXT,
  camera_model TEXT,
  content_category TEXT,
  processed_at TIMESTAMPTZ,
  summary_similarity FLOAT,
  keyword_similarity FLOAT,
  combined_similarity FLOAT
)
LANGUAGE SQL
AS $$
WITH summary_search AS (
  SELECT
    c.id, c.file_name, c.local_path, c.content_summary, 
    c.content_tags, c.duration_seconds, c.camera_make, c.camera_model,
    c.content_category, c.processed_at,
    (v.summary_vector <#> query_summary_embedding) * -1 as summary_similarity,
    ROW_NUMBER() OVER (ORDER BY v.summary_vector <#> query_summary_embedding) as rank_ix
  FROM clips c
  JOIN vectors v ON c.id = v.clip_id
  WHERE c.user_id = user_id_filter
    AND v.embedding_type = 'full_clip'
    AND v.summary_vector IS NOT NULL
  ORDER BY v.summary_vector <#> query_summary_embedding
  LIMIT LEAST(match_count * 2, 50)
),
keyword_search AS (
  SELECT
    c.id,
    (v.keyword_vector <#> query_keyword_embedding) * -1 as keyword_similarity
  FROM clips c
  JOIN vectors v ON c.id = v.clip_id
  WHERE c.user_id = user_id_filter
    AND v.embedding_type = 'full_clip'
    AND v.keyword_vector IS NOT NULL
  ORDER BY v.keyword_vector <#> query_keyword_embedding
  LIMIT LEAST(match_count * 2, 50)
)
SELECT
  ss.id,
  ss.file_name,
  ss.local_path,
  ss.content_summary,
  ss.content_tags,
  ss.duration_seconds,
  ss.camera_make,
  ss.camera_model,
  ss.content_category,
  ss.processed_at,
  ss.summary_similarity,
  COALESCE(ks.keyword_similarity, 0.0) as keyword_similarity,
  (ss.summary_similarity * summary_weight + COALESCE(ks.keyword_similarity, 0.0) * keyword_weight) as combined_similarity
FROM summary_search ss
LEFT JOIN keyword_search ks ON ss.id = ks.id
WHERE ss.summary_similarity >= similarity_threshold
ORDER BY combined_similarity DESC
LIMIT match_count;
$$;

-- Function for hybrid search combining full-text and semantic search using RRF
CREATE OR REPLACE FUNCTION hybrid_search_clips(
  query_text TEXT,
  query_summary_embedding vector(1024),
  query_keyword_embedding vector(1024),
  user_id_filter UUID,
  match_count INT DEFAULT 10,
  fulltext_weight FLOAT DEFAULT 1.0,
  summary_weight FLOAT DEFAULT 1.0,
  keyword_weight FLOAT DEFAULT 0.8,
  rrf_k INT DEFAULT 50
)
RETURNS TABLE (
  id UUID,
  file_name TEXT,
  local_path TEXT,
  content_summary TEXT,
  content_tags TEXT[],
  duration_seconds NUMERIC,
  camera_make TEXT,
  camera_model TEXT,
  content_category TEXT,
  processed_at TIMESTAMPTZ,
  transcript_preview TEXT,
  similarity_score FLOAT,
  search_rank FLOAT,
  match_type TEXT
)
LANGUAGE SQL
AS $$
WITH fulltext AS (
  SELECT
    c.id, c.file_name, c.local_path, c.content_summary, 
    c.content_tags, c.duration_seconds, c.camera_make, c.camera_model,
    c.content_category, c.processed_at, c.transcript_preview,
    ts_rank_cd(c.fts, websearch_to_tsquery('english', query_text)) as fts_score,
    ROW_NUMBER() OVER(ORDER BY ts_rank_cd(c.fts, websearch_to_tsquery('english', query_text)) DESC) as rank_ix
  FROM clips c
  WHERE c.user_id = user_id_filter
    AND c.fts @@ websearch_to_tsquery('english', query_text)
  ORDER BY ts_rank_cd(c.fts, websearch_to_tsquery('english', query_text)) DESC
  LIMIT LEAST(match_count * 2, 30)
),
summary_semantic AS (
  SELECT
    c.id, c.file_name, c.local_path, c.content_summary,
    c.content_tags, c.duration_seconds, c.camera_make, c.camera_model,
    c.content_category, c.processed_at, c.transcript_preview,
    (v.summary_vector <#> query_summary_embedding) * -1 as similarity_score,
    ROW_NUMBER() OVER (ORDER BY v.summary_vector <#> query_summary_embedding) as rank_ix
  FROM clips c
  JOIN vectors v ON c.id = v.clip_id
  WHERE c.user_id = user_id_filter
    AND v.embedding_type = 'full_clip'
    AND v.summary_vector IS NOT NULL
  ORDER BY v.summary_vector <#> query_summary_embedding
  LIMIT LEAST(match_count * 2, 30)
),
keyword_semantic AS (
  SELECT
    c.id,
    (v.keyword_vector <#> query_keyword_embedding) * -1 as keyword_similarity,
    ROW_NUMBER() OVER (ORDER BY v.keyword_vector <#> query_keyword_embedding) as rank_ix
  FROM clips c
  JOIN vectors v ON c.id = v.clip_id
  WHERE c.user_id = user_id_filter
    AND v.embedding_type = 'full_clip'
    AND v.keyword_vector IS NOT NULL
  ORDER BY v.keyword_vector <#> query_keyword_embedding
  LIMIT LEAST(match_count * 2, 30)
)
SELECT
  COALESCE(ft.id, ss.id) as id,
  COALESCE(ft.file_name, ss.file_name) as file_name,
  COALESCE(ft.local_path, ss.local_path) as local_path,
  COALESCE(ft.content_summary, ss.content_summary) as content_summary,
  COALESCE(ft.content_tags, ss.content_tags) as content_tags,
  COALESCE(ft.duration_seconds, ss.duration_seconds) as duration_seconds,
  COALESCE(ft.camera_make, ss.camera_make) as camera_make,
  COALESCE(ft.camera_model, ss.camera_model) as camera_model,
  COALESCE(ft.content_category, ss.content_category) as content_category,
  COALESCE(ft.processed_at, ss.processed_at) as processed_at,
  COALESCE(ft.transcript_preview, ss.transcript_preview) as transcript_preview,
  COALESCE(ss.similarity_score, 0.0) as similarity_score,
  -- RRF SCORING WITH DUAL VECTORS AND FULL-TEXT
  COALESCE(1.0 / (rrf_k + ft.rank_ix), 0.0) * fulltext_weight +
  COALESCE(1.0 / (rrf_k + ss.rank_ix), 0.0) * summary_weight +
  COALESCE(1.0 / (rrf_k + ks.rank_ix), 0.0) * keyword_weight as search_rank,
  CASE 
    WHEN ft.id IS NOT NULL AND ss.id IS NOT NULL THEN 'hybrid'
    WHEN ft.id IS NOT NULL THEN 'fulltext'
    ELSE 'semantic'
  END as match_type
FROM fulltext ft
FULL OUTER JOIN summary_semantic ss ON ft.id = ss.id
FULL OUTER JOIN keyword_semantic ks ON COALESCE(ft.id, ss.id) = ks.id
ORDER BY search_rank DESC
LIMIT match_count;
$$;

-- Function for full-text search only
CREATE OR REPLACE FUNCTION fulltext_search_clips(
  query_text TEXT,
  user_id_filter UUID,
  match_count INT DEFAULT 10
)
RETURNS TABLE (
  id UUID,
  file_name TEXT,
  local_path TEXT,
  content_summary TEXT,
  content_tags TEXT[],
  duration_seconds NUMERIC,
  camera_make TEXT,
  camera_model TEXT,
  content_category TEXT,
  processed_at TIMESTAMPTZ,
  transcript_preview TEXT,
  fts_rank FLOAT
)
LANGUAGE SQL
AS $$
SELECT
  c.id,
  c.file_name,
  c.local_path,
  c.content_summary,
  c.content_tags,
  c.duration_seconds,
  c.camera_make,
  c.camera_model,
  c.content_category,
  c.processed_at,
  c.transcript_preview,
  ts_rank_cd(c.fts, websearch_to_tsquery('english', query_text)) as fts_rank
FROM clips c
WHERE c.user_id = user_id_filter
  AND c.fts @@ websearch_to_tsquery('english', query_text)
ORDER BY ts_rank_cd(c.fts, websearch_to_tsquery('english', query_text)) DESC
LIMIT match_count;
$$;

-- Function to search transcripts specifically
CREATE OR REPLACE FUNCTION search_transcripts(
  query_text TEXT,
  user_id_filter UUID,
  match_count INT DEFAULT 10,
  min_content_length INT DEFAULT 50
)
RETURNS TABLE (
  clip_id UUID,
  file_name TEXT,
  local_path TEXT,
  content_summary TEXT,
  full_text TEXT,
  transcript_preview TEXT,
  duration_seconds NUMERIC,
  processed_at TIMESTAMPTZ,
  fts_rank FLOAT
)
LANGUAGE SQL
AS $$
SELECT
  t.clip_id,
  c.file_name,
  c.local_path,
  c.content_summary,
  t.full_text,
  c.transcript_preview,
  c.duration_seconds,
  c.processed_at,
  ts_rank_cd(t.fts, websearch_to_tsquery('english', query_text)) as fts_rank
FROM transcripts t
JOIN clips c ON t.clip_id = c.id
WHERE t.user_id = user_id_filter
  AND LENGTH(t.full_text) >= min_content_length
  AND t.fts @@ websearch_to_tsquery('english', query_text)
ORDER BY ts_rank_cd(t.fts, websearch_to_tsquery('english', query_text)) DESC
LIMIT match_count;
$$;

-- Function to find similar clips based on existing clip
CREATE OR REPLACE FUNCTION find_similar_clips(
  source_clip_id UUID,
  user_id_filter UUID,
  match_count INT DEFAULT 5,
  similarity_threshold FLOAT DEFAULT 0.5
)
RETURNS TABLE (
  id UUID,
  file_name TEXT,
  local_path TEXT,
  content_summary TEXT,
  content_tags TEXT[],
  duration_seconds NUMERIC,
  content_category TEXT,
  similarity_score FLOAT
)
LANGUAGE SQL
AS $$
WITH source_vector AS (
  SELECT v.summary_vector
  FROM vectors v
  WHERE v.clip_id = source_clip_id
    AND v.embedding_type = 'full_clip'
    AND v.summary_vector IS NOT NULL
  LIMIT 1
)
SELECT
  c.id,
  c.file_name,
  c.local_path,
  c.content_summary,
  c.content_tags,
  c.duration_seconds,
  c.content_category,
  (v.summary_vector <#> sv.summary_vector) * -1 as similarity_score
FROM clips c
JOIN vectors v ON c.id = v.clip_id
CROSS JOIN source_vector sv
WHERE c.user_id = user_id_filter
  AND c.id != source_clip_id
  AND v.embedding_type = 'full_clip'
  AND v.summary_vector IS NOT NULL
  AND (v.summary_vector <#> sv.summary_vector) * -1 >= similarity_threshold
ORDER BY v.summary_vector <#> sv.summary_vector
LIMIT match_count;
$$;
```

### 10. Registry System (`core/registry.py`)

```python
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Type, Optional

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepConfig

class StepRegistry:
    """Auto-discovers and manages all available steps"""
    
    def __init__(self):
        self.steps: Dict[str, Dict[str, Type[BaseStep]]] = {}
        self._discovered = False
    
    def discover_steps(self):
        """Auto-discover all steps in the steps directory"""
        if self._discovered:
            return
            
        steps_dir = Path(__file__).parent.parent / "steps"
        
        # Iterate through category directories
        for category_dir in steps_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith("_"):
                continue
                
            category = category_dir.name
            self.steps[category] = {}
            
            # Import all Python files in the category
            for module_file in category_dir.glob("*.py"):
                if module_file.name.startswith("_"):
                    continue
                
                try:
                    # Import the module
                    module_name = f"video_tool.steps.{category}.{module_file.stem}"
                    module = importlib.import_module(module_name)
                    
                    # Find all BaseStep subclasses
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseStep) and 
                            obj != BaseStep and
                            hasattr(obj, 'name') and
                            obj.name):  # Ensure it has a name
                            
                            self.steps[category][obj.name] = obj
                            
                except Exception as e:
                    print(f"Error loading step module {module_name}: {e}")
        
        self._discovered = True
    
    def get_step_class(self, category: str, name: str) -> Optional[Type[BaseStep]]:
        """Get a specific step class"""
        self.discover_steps()
        return self.steps.get(category, {}).get(name)
    
    def create_step(self, category: str, name: str, config: StepConfig) -> Optional[BaseStep]:
        """Create a step instance"""
        step_class = self.get_step_class(category, name)
        if step_class:
            return step_class(config)
        return None
    
    def list_steps(self) -> Dict[str, List[Dict[str, str]]]:
        """List all available steps with their info"""
        self.discover_steps()
        
        result = {}
        for category, steps in self.steps.items():
            result[category] = []
            for name, step_class in steps.items():
                # Create temporary instance to get info
                temp_step = step_class(StepConfig())
                result[category].append({
                    "name": name,
                    "version": temp_step.version,
                    "description": temp_step.description
                })
        
        return result
    
    def get_categories(self) -> List[str]:
        """Get all step categories"""
        self.discover_steps()
        return list(self.steps.keys())
```

### 11. Pipeline Orchestrator with Incremental Saves (`core/pipeline.py`) - UPDATED

```python
import yaml
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import asyncio
from datetime import datetime, timedelta
import structlog

from video_tool.core.models import StepConfig, VideoMetadata, PipelineConfig, ProcessingStatus
from video_tool.core.registry import StepRegistry
from video_tool.core.db import get_procrastinate_app, get_supabase
from video_tool.steps.base import BaseStep

logger = structlog.get_logger(__name__)

class Pipeline:
    """Orchestrates step execution with incremental saves and monitoring"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.registry = StepRegistry()
        self.steps: List[BaseStep] = []
        self.app = None  # Procrastinate app
        self.supabase = None
        self._initialized = False
    
    def _load_config(self) -> PipelineConfig:
        """Load pipeline configuration with environment variable substitution"""
        with open(self.config_path) as f:
            content = f.read()
            
            # Substitute environment variables ${VAR_NAME}
            for match in re.findall(r'\${(\w+)}', content):
                value = os.getenv(match, '')
                content = content.replace(f'${{{match}}}', value)
            
            data = yaml.safe_load(content)
            return PipelineConfig(**data)
    
    async def initialize(self):
        """Initialize pipeline components"""
        if self._initialized:
            return
            
        # Get Procrastinate app and Supabase client
        self.app = await get_procrastinate_app()
        self.supabase = await get_supabase()
        
        # Initialize steps from config
        for step_config in self.config.steps:
            config = StepConfig(**step_config.get('config', {}))
            
            step = self.registry.create_step(
                step_config['category'],
                step_config['name'],
                config
            )
            
            if not step:
                raise ValueError(
                    f"Step not found: {step_config['category']}.{step_config['name']}"
                )
            
            await step.setup()
            self.steps.append(step)
        
        # UPDATED: Register Procrastinate tasks using current patterns
        await self._register_tasks()
        
        # Register monitoring tasks  
        await self._register_monitoring_tasks()
        
        self._initialized = True
    
    async def _register_tasks(self):
        """UPDATED: Register all steps as Procrastinate tasks using current patterns"""
        for i, step in enumerate(self.steps):
            
            # UPDATED: Use current task registration pattern
            @self.app.task(
                name=f"video_tool.steps.{step.category}.{step.name}",
                queue=step.config.queue,
                retry=step.config.retry,
                timeout=step.config.timeout,
                pass_context=True  # UPDATED: Use pass_context for better error handling
            )
            async def process_step_task(context, video_id: str, step_idx: int = i):
                """UPDATED: Task function with context support"""
                try:
                    # Get current video metadata
                    video = await self._get_video_metadata(video_id)
                    if not video:
                        raise ValueError(f"Video not found: {video_id}")
                    
                    current_step = self.steps[step_idx]
                    
                    # Check if already completed (for resume functionality)
                    if current_step.name in video.processed_steps:
                        logger.info(f"Step {current_step.name} already completed, skipping")
                        # Queue next step
                        await self._queue_next_step(video_id, step_idx)
                        return {"status": "skipped", "reason": "already_completed"}
                    
                    # UPDATED: Execute step with context
                    result = await current_step.execute(
                        video, 
                        step_idx + 1,  # Human-readable step number
                        len(self.steps),
                        context.additional_context  # Pass Procrastinate context
                    )
                    
                    if result.success:
                        # Mark step as completed
                        await self._mark_step_completed(video_id, current_step.name)
                        
                        # Queue next step or mark pipeline complete
                        if step_idx + 1 < len(self.steps):
                            await self._queue_next_step(video_id, step_idx)
                        else:
                            await self._mark_pipeline_completed(video_id)
                    else:
                        # Handle failure
                        await self._mark_step_failed(video_id, current_step.name, result.error)
                        
                        # Decide whether to continue or stop
                        if current_step.config.params.get('continue_on_failure', False):
                            await self._queue_next_step(video_id, step_idx)
                    
                    return result.dict()
                    
                except Exception as e:
                    logger.error(f"Task execution failed: {str(e)}", exc_info=True)  
                    await self._mark_step_failed(video_id, self.steps[step_idx].name, str(e))
                    raise  # Let Procrastinate handle retry logic
            
            # Store task reference on the step
            step.task = process_step_task
    
    async def _register_monitoring_tasks(self):
        """Register periodic monitoring tasks"""
        
        @self.app.periodic(cron="*/5 * * * *")  # Every 5 minutes
        async def check_stalled_videos():
            """Find and restart stalled video processing"""
            stalled = await self.supabase.table('clips')\
                .select('id, current_step, updated_at')\
                .lt('processing_progress', 100)\
                .lt('updated_at', (datetime.utcnow() - timedelta(minutes=30)).isoformat())\
                .execute()
            
            for video in stalled.data:
                self.logger.warning(f"Found stalled video: {video['id']}")
                await self.resume_video(video['id'])
        
        @self.app.periodic(cron="0 * * * *")  # Every hour
        async def cleanup_old_artifacts():
            """Clean up artifacts from completed videos older than 7 days"""
            old_artifacts = await self.supabase.table('artifacts')\
                .select('*')\
                .lt('created_at', (datetime.utcnow() - timedelta(days=7)).isoformat())\
                .execute()
            
            for artifact in old_artifacts.data:
                # Delete file if exists
                file_path = Path(artifact['file_path'])
                if file_path.exists():
                    file_path.unlink()
                
                # Remove record
                await self.supabase.table('artifacts')\
                    .delete()\
                    .eq('id', artifact['id'])\
                    .execute()
    
    async def process_video(self, file_path: str, auth_manager: 'AuthManager') -> str:
        """Start processing a video through the pipeline with authentication"""
        if not self._initialized:
            await self.initialize()
        
        # Get authenticated user
        session = auth_manager.get_current_session()
        if not session:
            raise ValueError("Authentication required")
        
        user_id = session['user_id']
        
        # Check for existing video with same path
        existing = await self.supabase.table('clips')\
            .select('id, checksum, status')\
            .eq('file_path', file_path)\
            .eq('user_id', user_id)\
            .execute()
        
        if existing.data:
            video_id = existing.data[0]['id']
            status = existing.data[0]['status']
            
            if status == ProcessingStatus.COMPLETED:
                self.logger.info(f"Video already processed: {video_id}")
                return video_id
            else:
                self.logger.info(f"Resuming processing for video: {video_id}")
                await self.resume_video(video_id)
                return video_id
        
        # Create initial video record
        from uuid import uuid4
        video = VideoMetadata(
            video_id=str(uuid4()),
            file_path=file_path,
            file_name=Path(file_path).name,
            user_id=user_id,
            status=ProcessingStatus.QUEUED,
            total_steps=len([s for s in self.steps if s.config.enabled])
        )
        
        # Store in database
        await self._create_video_record(video)
        
        # Queue first enabled step
        await self._queue_first_step(video.video_id)
        
        return video.video_id
    
    async def resume_video(self, video_id: str):
        """Resume processing from last completed step"""
        video = await self._get_video_metadata(video_id)
        if not video:
            return
        
        # Find last completed step
        last_completed_idx = -1
        for i, step in enumerate(self.steps):
            if step.name in video.processed_steps:
                last_completed_idx = i
        
        # Queue next step
        next_idx = last_completed_idx + 1
        if next_idx < len(self.steps):
            await self.steps[next_idx].task.defer_async(
                video_id=video_id,
                step_idx=next_idx
            )
    
    async def _queue_first_step(self, video_id: str):
        """Queue the first enabled step"""
        for i, step in enumerate(self.steps):
            if step.config.enabled:
                await step.task.defer_async(
                    video_id=video_id,
                    step_idx=i
                )
                break
    
    async def _queue_next_step(self, video_id: str, current_idx: int):
        """Queue the next enabled step"""
        for i in range(current_idx + 1, len(self.steps)):
            if self.steps[i].config.enabled:
                await self.steps[i].task.defer_async(
                    video_id=video_id,
                    step_idx=i
                )
                break
    
    async def _get_video_metadata(self, video_id: str) -> Optional[VideoMetadata]:
        """Get current video metadata from database"""
        result = await self.supabase.table('clips')\
            .select('*')\
            .eq('id', video_id)\
            .single()\
            .execute()
        
        if result.data:
            return VideoMetadata(**result.data)
        return None
    
    async def _create_video_record(self, video: VideoMetadata):
        """Create initial video record"""
        await self.supabase.table('clips').insert(video.dict()).execute()
    
    async def _mark_step_completed(self, video_id: str, step_name: str):
        """Mark a step as completed"""
        # Get current processed steps
        result = await self.supabase.table('clips')\
            .select('processed_steps')\
            .eq('id', video_id)\
            .single()\
            .execute()
        
        processed_steps = result.data.get('processed_steps', [])
        if step_name not in processed_steps:
            processed_steps.append(step_name)
        
        await self.supabase.table('clips').update({
            'processed_steps': processed_steps,
            'last_step_completed': step_name
        }).eq('id', video_id).execute()
    
    async def _mark_step_failed(self, video_id: str, step_name: str, error: str):
        """Mark a step as failed"""
        await self.supabase.table('clips').update({
            'status': ProcessingStatus.FAILED,
            'current_step': step_name,
            'error': f"{step_name}: {error}"
        }).eq('id', video_id).execute()
    
    async def _mark_pipeline_completed(self, video_id: str):
        """Mark entire pipeline as completed"""
        await self.supabase.table('clips').update({
            'status': ProcessingStatus.COMPLETED,
            'processing_progress': 100,
            'completed_at': datetime.utcnow().isoformat()
        }).eq('id', video_id).execute()

class PipelineMonitor:
    """Real-time pipeline monitoring"""
    
    def __init__(self, supabase):
        self.supabase = supabase
    
    async def get_active_videos(self) -> List[Dict]:
        """Get all actively processing videos"""
        result = await self.supabase.table('active_processing')\
            .select('*')\
            .execute()
        return result.data
    
    async def get_video_progress(self, video_id: str) -> Dict:
        """Get detailed progress for a specific video"""
        video = await self.supabase.table('clips')\
            .select('*, processing_events(*)')\
            .eq('id', video_id)\
            .single()\
            .execute()
        
        if not video.data:
            return {}
        
        # Calculate step timings
        events = video.data.get('processing_events', [])
        step_timings = {}
        
        for event in sorted(events, key=lambda e: e['timestamp']):
            step = event['step_name']
            if step not in step_timings:
                step_timings[step] = {}
            
            if event['status'] == 'starting':
                step_timings[step]['start'] = event['timestamp']
            elif event['status'] == 'completed':
                step_timings[step]['end'] = event['timestamp']
                if 'start' in step_timings[step]:
                    start = datetime.fromisoformat(step_timings[step]['start'])
                    end = datetime.fromisoformat(step_timings[step]['end'])
                    step_timings[step]['duration'] = (end - start).total_seconds()
        
        return {
            'video': video.data,
            'step_timings': step_timings,
            'total_duration': sum(s.get('duration', 0) for s in step_timings.values())
        }
    
    async def watch_video(self, video_id: str, callback=None):
        """Watch a video's progress in real-time"""
        last_progress = -1
        
        while True:
            progress = await self.get_video_progress(video_id)
            current_progress = progress['video'].get('processing_progress', 0)
            
            if current_progress != last_progress:
                if callback:
                    await callback(progress)
                else:
                    print(f"Progress: {current_progress}% - {progress['video'].get('current_step')}")
                
                last_progress = current_progress
            
            if current_progress >= 100 or progress['video'].get('status') == 'failed':
                break
            
            await asyncio.sleep(2)
```

### 12. CLI Implementation with Authentication (`cli.py`)

```python
import asyncio
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
import yaml

from video_tool.core.pipeline import Pipeline
from video_tool.core.registry import StepRegistry
from video_tool.core.db import get_procrastinate_app, get_supabase
from video_tool.core.auth import AuthManager
from video_tool.core.search import VideoSearcher, format_search_results
from video_tool.worker import run_workers

app = typer.Typer(help="Modular Video Processing Tool")
console = Console()

# Create sub-apps
auth_app = typer.Typer(help="Authentication commands")
search_app = typer.Typer(help="Search video catalog")
app.add_typer(auth_app, name="auth")
app.add_typer(search_app, name="search")

@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Directory or file to process"),
    config: str = typer.Option("default", "--config", "-c", help="Pipeline configuration"),
    workers: int = typer.Option(0, "--workers", "-w", help="Number of workers to run"),
    watch: bool = typer.Option(False, "--watch", help="Watch directory for new files"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-nr"),
    pattern: str = typer.Option("*.mp4,*.mov,*.avi", "--pattern", "-p"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be processed")
):
    """Ingest videos using configured pipeline (requires authentication)"""
    
    async def run():
        # Check authentication
        auth_manager = AuthManager()
        session = auth_manager.get_current_session()
        if not session:
            console.print("[red]Authentication required. Please login first.[/red]")
            console.print("Run: [cyan]video-tool auth login[/cyan]")
            return
        
        console.print(f"[green]Authenticated as: {session['email']}[/green]\n")
        
        # Load pipeline
        config_path = Path("configs") / f"{config}.yaml"
        if not config_path.exists():
            console.print(f"[red]Configuration not found: {config_path}[/red]")
            return
        
        pipeline = Pipeline(str(config_path))
        await pipeline.initialize()
        
        # Show pipeline info
        console.print(f"[cyan]Using pipeline: {pipeline.config.name}[/cyan]")
        console.print(f"[dim]{pipeline.config.description}[/dim]\n")
        
        # Show steps
        steps_table = Table(title="Pipeline Steps")
        steps_table.add_column("Order", style="cyan")
        steps_table.add_column("Category", style="magenta")
        steps_table.add_column("Step", style="green")
        steps_table.add_column("Status", style="yellow")
        steps_table.add_column("Queue", style="blue")
        
        for i, step in enumerate(pipeline.steps):
            status = "✓ Enabled" if step.config.enabled else "✗ Disabled"
            steps_table.add_row(
                str(i + 1),
                step.category,
                step.name,
                status,
                step.config.queue
            )
        
        console.print(steps_table)
        
        if dry_run:
            console.print("\n[yellow]Dry run mode - not processing files[/yellow]")
            return
        
        # Find video files
        video_files = find_video_files(path, recursive, pattern)
        
        if not video_files:
            console.print("[yellow]No video files found[/yellow]")
            return
        
        console.print(f"\n[green]Found {len(video_files)} video files[/green]")
        
        # Process files
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Queueing videos...", total=len(video_files))
            
            video_ids = []
            for file_path in video_files:
                video_id = await pipeline.process_video(str(file_path), auth_manager)
                video_ids.append(video_id)
                progress.advance(task)
        
        console.print(f"[green]✓ Queued {len(video_ids)} videos for processing[/green]")
        
        # Start workers if requested
        if workers > 0:
            console.print(f"\n[cyan]Starting {workers} workers...[/cyan]")
            
            app = await get_procrastinate_app()
            
            # Run workers
            await run_workers(app, workers, pipeline.config.worker_config)
        else:
            console.print("\n[dim]Run workers separately with: video-tool worker[/dim]")
        
        # If watching, start watcher
        if watch:
            console.print(f"\n[cyan]Watching {path} for new files...[/cyan]")
            await watch_directory(path, pipeline, pattern, recursive, auth_manager)
    
    # Run async function
    asyncio.run(run())

# Authentication commands
@auth_app.command("login")
def auth_login(
    email: Optional[str] = typer.Option(None, "--email", "-e", help="Email address"),
    password: Optional[str] = typer.Option(None, "--password", "-p", help="Password")
):
    """Login to your account"""
    
    async def run():
        auth_manager = AuthManager()
        
        # Check if already logged in
        session = auth_manager.get_current_session()
        if session:
            console.print(f"[yellow]Already logged in as: {session['email']}[/yellow]")
            if not typer.confirm("Do you want to login with a different account?"):
                return
        
        # Get credentials
        if not email:
            email = Prompt.ask("Email")
        if not password:
            password = Prompt.ask("Password", password=True)
        
        # Attempt login
        with console.status("[bold green]Logging in..."):
            success = await auth_manager.login(email, password)
        
        if success:
            console.print("[green]✓ Successfully logged in![/green]")
            
            # Get and show user profile
            profile = await auth_manager.get_user_profile()
            if profile:
                console.print(f"Welcome, {profile.get('display_name', email)}!")
                if profile.get('profile_type') == 'admin':
                    console.print("[yellow]Admin access granted[/yellow]")
        else:
            console.print("[red]✗ Login failed. Please check your credentials.[/red]")
    
    asyncio.run(run())

@auth_app.command("signup")
def auth_signup(
    email: Optional[str] = typer.Option(None, "--email", "-e", help="Email address"),
    password: Optional[str] = typer.Option(None, "--password", "-p", help="Password")
):
    """Create a new account"""
    
    async def run():
        auth_manager = AuthManager()
        
        # Get credentials
        if not email:
            email = Prompt.ask("Email")
        if not password:
            password = Prompt.ask("Password", password=True)
            confirm_password = Prompt.ask("Confirm Password", password=True)
            
            if password != confirm_password:
                console.print("[red]Passwords do not match![/red]")
                return
        
        # Attempt signup
        with console.status("[bold green]Creating account..."):
            success = await auth_manager.signup(email, password)
        
        if success:
            console.print("[green]✓ Account created successfully![/green]")
            console.print("[yellow]Please check your email to confirm your account.[/yellow]")
        else:
            console.print("[red]✗ Signup failed. Email may already be registered.[/red]")
    
    asyncio.run(run())

@auth_app.command("logout")
def auth_logout():
    """Logout from current session"""
    
    async def run():
        auth_manager = AuthManager()
        
        session = auth_manager.get_current_session()
        if not session:
            console.print("[yellow]Not currently logged in[/yellow]")
            return
        
        if typer.confirm(f"Logout from {session['email']}?"):
            success = await auth_manager.logout()
            if success:
                console.print("[green]✓ Successfully logged out[/green]")
            else:
                console.print("[red]✗ Logout failed[/red]")
    
    asyncio.run(run())

@auth_app.command("status")
def auth_status():
    """Show current authentication status"""
    
    async def run():
        auth_manager = AuthManager()
        
        session = auth_manager.get_current_session()
        if not session:
            console.print("[yellow]Not logged in[/yellow]")
            console.print("Run: [cyan]video-tool auth login[/cyan]")
            return
        
        # Show session info
        status_table = Table(title="Authentication Status")
        status_table.add_column("Property", style="cyan")
        status_table.add_column("Value", style="green")
        
        status_table.add_row("Email", session['email'])
        status_table.add_row("User ID", session['user_id'][:8] + "...")
        
        # Get profile info
        profile = await auth_manager.get_user_profile()
        if profile:
            status_table.add_row("Display Name", profile.get('display_name', 'N/A'))
            status_table.add_row("Account Type", profile.get('profile_type', 'user'))
            status_table.add_row("Created", profile.get('created_at', 'N/A')[:10])
        
        console.print(status_table)
        
        # Get user stats
        client = await auth_manager.get_authenticated_client()
        if client:
            stats_result = client.rpc('get_user_stats').execute()
            if stats_result.data:
                stats = stats_result.data
                
                stats_table = Table(title="Video Processing Statistics")
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="yellow")
                
                stats_table.add_row("Total Videos", str(stats.get('total_clips', 0)))
                stats_table.add_row("Total Duration", f"{stats.get('total_duration_hours', 0):.1f} hours")
                stats_table.add_row("Storage Used", f"{stats.get('total_storage_gb', 0):.2f} GB")
                stats_table.add_row("Videos with Transcripts", str(stats.get('clips_with_transcripts', 0)))
                stats_table.add_row("In Progress", str(stats.get('clips_in_progress', 0)))
                stats_table.add_row("Completed", str(stats.get('clips_completed', 0)))
                
                console.print("\n")
                console.print(stats_table)
    
    asyncio.run(run())

# Search commands
@search_app.command("query")
def search_query(
    query: str = typer.Argument(..., help="Search query"),
    search_type: str = typer.Option("hybrid", "--type", "-t", help="Search type"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results"),
    show_scores: bool = typer.Option(False, "--scores", help="Show similarity scores")
):
    """Search the video catalog using various search methods"""
    
    async def run():
        # Validate search type
        valid_types = ["semantic", "fulltext", "hybrid", "transcripts"]
        if search_type not in valid_types:
            console.print(f"[red]Invalid search type. Must be one of: {', '.join(valid_types)}[/red]")
            return
        
        searcher = VideoSearcher()
        
        # Set weights for search
        weights = {
            'summary': 1.0,
            'keyword': 0.8,
            'fulltext': 1.0,
            'threshold': 0.3
        }
        
        try:
            # Perform search
            results = await searcher.search(
                query=query,
                search_type=search_type,
                match_count=limit,
                weights=weights
            )
            
            # Format results
            formatted_results = format_search_results(results, search_type, show_scores)
            
            if not formatted_results:
                console.print("[yellow]No results found[/yellow]")
                return
            
            # Display as table
            results_table = Table(title=f"Search Results ({len(results)} found)")
            results_table.add_column("File", style="cyan", max_width=30)
            results_table.add_column("Summary", style="green", max_width=50)
            results_table.add_column("Category", style="magenta")
            results_table.add_column("Duration", style="yellow")
            results_table.add_column("Camera", style="blue", max_width=20)
            
            if show_scores:
                if search_type == "semantic":
                    results_table.add_column("Similarity", style="red")
                elif search_type == "hybrid":
                    results_table.add_column("Rank", style="red")
                    results_table.add_column("Type", style="red")
                elif search_type == "fulltext":
                    results_table.add_column("FTS Rank", style="red")
            
            for result in formatted_results:
                row = [
                    result.get('file_name', 'Unknown'),
                    result.get('content_summary', 'No summary')[:47] + "..." if len(result.get('content_summary', '')) > 50 else result.get('content_summary', 'No summary'),
                    result.get('content_category', 'Unknown'),
                    result.get('duration', 'Unknown'),
                    result.get('camera', 'Unknown')
                ]
                
                if show_scores:
                    if search_type == "semantic":
                        row.append(result.get('similarity_score', '0.000'))
                    elif search_type == "hybrid":
                        row.append(result.get('search_rank', '0.000'))
                        row.append(result.get('match_type', 'unknown'))
                    elif search_type == "fulltext":
                        row.append(result.get('fts_rank', '0.000'))
                
                results_table.add_row(*row)
            
            console.print(results_table)
            
        except Exception as e:
            console.print(f"[red]Search failed: {str(e)}[/red]")
    
    asyncio.run(run())

@search_app.command("similar")
def search_similar(
    clip_id: str = typer.Argument(..., help="Clip ID to find similar videos"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of similar clips"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Similarity threshold")
):
    """Find videos similar to a given clip"""
    
    async def run():
        searcher = VideoSearcher()
        
        try:
            results = await searcher.find_similar(
                clip_id=clip_id,
                match_count=limit,
                similarity_threshold=threshold
            )
            
            formatted_results = format_search_results(results, "similar", True)
            
            if not formatted_results:
                console.print("[yellow]No similar clips found[/yellow]")
                return
            
            results_table = Table(title=f"Similar Clips ({len(results)} found)")
            results_table.add_column("File", style="cyan", max_width=30)
            results_table.add_column("Summary", style="green", max_width=50)
            results_table.add_column("Category", style="magenta")
            results_table.add_column("Duration", style="yellow")
            results_table.add_column("Similarity", style="red")
            
            for result in formatted_results:
                results_table.add_row(
                    result.get('file_name', 'Unknown'),
                    result.get('content_summary', 'No summary')[:47] + "..." if len(result.get('content_summary', '')) > 50 else result.get('content_summary', 'No summary'),
                    result.get('content_category', 'Unknown'),
                    result.get('duration', 'Unknown'),
                    f"{result.get('similarity_score', 0):.3f}"
                )
            
            console.print(results_table)
            
        except Exception as e:
            console.print(f"[red]Similar search failed: {str(e)}[/red]")
    
    asyncio.run(run())

@search_app.command("info")
def search_info(
    clip_id: str = typer.Argument(..., help="Clip ID to show details"),
    show_transcript: bool = typer.Option(False, "--transcript", help="Show full transcript"),
    show_analysis: bool = typer.Option(False, "--analysis", help="Show AI analysis")
):
    """Show detailed information about a specific clip"""
    
    async def run():
        searcher = VideoSearcher()
        
        try:
            clip = await searcher.get_clip_details(clip_id)
            
            if not clip:
                console.print(f"[red]Clip not found: {clip_id}[/red]")
                return
            
            # Display clip information
            info_table = Table(title=f"Clip Details: {clip.get('file_name')}")
            info_table.add_column("Property", style="cyan")
            info_table.add_column("Value", style="green")
            
            info_table.add_row("ID", clip.get('id', 'Unknown'))
            info_table.add_row("File Name", clip.get('file_name', 'Unknown'))
            info_table.add_row("Duration", format_duration(clip.get('duration_seconds', 0)))
            info_table.add_row("Size", format_file_size(clip.get('file_size_bytes', 0)))
            info_table.add_row("Resolution", f"{clip.get('width', '?')}x{clip.get('height', '?')}")
            info_table.add_row("FPS", str(clip.get('frame_rate', 'Unknown')))
            info_table.add_row("Codec", clip.get('codec', 'Unknown'))
            info_table.add_row("Camera", f"{clip.get('camera_make', '')} {clip.get('camera_model', '')}".strip() or 'Unknown')
            info_table.add_row("Category", clip.get('content_category', 'Unknown'))
            info_table.add_row("Processed", clip.get('processed_at', 'Unknown')[:19] if clip.get('processed_at') else 'Unknown')
            
            console.print(info_table)
            
            # Show content summary
            if clip.get('content_summary'):
                console.print(f"\n[bold cyan]Summary:[/bold cyan]")
                console.print(clip['content_summary'])
            
            # Show content tags
            if clip.get('content_tags'):
                console.print(f"\n[bold cyan]Tags:[/bold cyan]")
                console.print(" ".join([f"[blue]#{tag}[/blue]" for tag in clip['content_tags']]))
            
            # Show transcript if requested
            if show_transcript and clip.get('transcript'):
                console.print(f"\n[bold cyan]Transcript:[/bold cyan]")
                console.print(clip['transcript'])
            
            # Show AI analysis if requested
            if show_analysis and clip.get('ai_analysis'):
                console.print(f"\n[bold cyan]AI Analysis:[/bold cyan]")
                console.print(clip['ai_analysis'])
                
        except Exception as e:
            console.print(f"[red]Failed to get clip info: {str(e)}[/red]")
    
    asyncio.run(run())

@search_app.command("stats")
def show_catalog_stats():
    """Show statistics about your video catalog"""
    
    async def run():
        searcher = VideoSearcher()
        
        try:
            stats = await searcher.get_user_stats()
            
            if not stats:
                console.print("[yellow]No statistics available[/yellow]")
                return
            
            stats_table = Table(title="Video Catalog Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")
            
            stats_table.add_row("Total Videos", str(stats.get('total_clips', 0)))
            stats_table.add_row("Total Duration", f"{stats.get('total_duration_hours', 0):.1f} hours")
            stats_table.add_row("Storage Used", f"{stats.get('total_storage_gb', 0):.2f} GB")
            stats_table.add_row("With Transcripts", str(stats.get('clips_with_transcripts', 0)))
            stats_table.add_row("In Progress", str(stats.get('clips_in_progress', 0)))
            stats_table.add_row("Completed", str(stats.get('clips_completed', 0)))
            
            console.print(stats_table)
            
        except Exception as e:
            console.print(f"[red]Failed to get stats: {str(e)}[/red]")
    
    asyncio.run(run())

@app.command()
def worker(
    workers: int = typer.Option(4, "--workers", "-w", help="Number of workers"),
    queues: List[str] = typer.Option(None, "--queue", "-q", help="Specific queues to process"),
    config: str = typer.Option("default", "--config", "-c", help="Pipeline configuration")
):
    """Run Procrastinate workers"""
    
    async def run():
        # Load pipeline config to get worker settings
        config_path = Path("configs") / f"{config}.yaml"
        pipeline = Pipeline(str(config_path))
        await pipeline.initialize()
        
        app = await get_procrastinate_app()
        
        # Determine queues
        if not queues:
            queues = list(pipeline.config.worker_config.keys())
        
        console.print(f"[cyan]Starting {workers} workers for queues: {', '.join(queues)}[/cyan]")
        
        await run_workers(app, workers, pipeline.config.worker_config, queues)
    
    asyncio.run(run())

@app.command()
def status(
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch status continuously")
):
    """Show processing status"""
    
    async def show_status():
        # Check authentication
        auth_manager = AuthManager()
        client = await auth_manager.get_authenticated_client()
        if not client:
            console.print("[red]Authentication required. Please login first.[/red]")
            return
        
        app = await get_procrastinate_app()
        
        while True:
            # Get queue stats from Procrastinate
            jobs = await app.jobs.count_by_status()
            
            # Get user's video stats from Supabase
            videos_result = client.table('clips')\
                .select('status')\
                .eq('user_id', auth_manager.get_current_session()['user_id'])\
                .execute()
                
            video_stats = {}
            for video in videos_result.data:
                status = video['status']
                video_stats[status] = video_stats.get(status, 0) + 1
            
            # Clear screen if watching
            if watch:
                console.clear()
            
            # Show job queue status
            queue_table = Table(title="Job Queue Status")
            queue_table.add_column("Status", style="cyan")
            queue_table.add_column("Count", style="yellow")
            
            for status, count in jobs.items():
                queue_table.add_row(status, str(count))
            
            console.print(queue_table)
            
            # Show video status
            video_table = Table(title="Your Video Processing Status")
            video_table.add_column("Status", style="cyan")
            video_table.add_column("Count", style="yellow")
            
            for status, count in sorted(video_stats.items()):
                video_table.add_row(status, str(count))
            
            console.print(video_table)
            
            if not watch:
                break
            
            await asyncio.sleep(2)
    
    asyncio.run(show_status())

@app.command()
def monitor(
    video_id: Optional[str] = typer.Option(None, "--video", "-v", help="Monitor specific video"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow progress in real-time")
):
    """Monitor processing progress"""
    
    async def run_monitor():
        from video_tool.core.pipeline import PipelineMonitor
        
        monitor = PipelineMonitor(await get_supabase())
        
        if video_id:
            # Monitor specific video
            if follow:
                # Real-time following
                async def progress_callback(progress):
                    console.clear()
                    
                    # Show video info
                    video = progress['video']
                    info_table = Table(title=f"Video: {video['file_name']}")
                    info_table.add_column("Property", style="cyan")
                    info_table.add_column("Value", style="green")
                    
                    info_table.add_row("Status", video['status'])
                    info_table.add_row("Progress", f"{video['processing_progress']}%")
                    info_table.add_row("Current Step", video.get('current_step', 'N/A'))
                    info_table.add_row("Steps Completed", f"{len(video.get('processed_steps', []))}/{video.get('total_steps', 0)}")
                    
                    console.print(info_table)
                    
                    # Show step timings
                    if progress['step_timings']:
                        timing_table = Table(title="Step Timings")
                        timing_table.add_column("Step", style="cyan")
                        timing_table.add_column("Duration", style="yellow")
                        timing_table.add_column("Status", style="green")
                        
                        for step, timing in progress['step_timings'].items():
                            duration = f"{timing.get('duration', 0):.1f}s" if 'duration' in timing else "Running..."
                            status = "✓" if 'end' in timing else "⏳"
                            timing_table.add_row(step, duration, status)
                        
                        console.print(timing_table)
                    
                    # Show available features
                    features_table = Table(title="Available Features")
                    features_table.add_column("Feature", style="cyan")
                    features_table.add_column("Status", style="green")
                    
                    features_table.add_row("Metadata", "✓" if video.get('duration_seconds') else "✗")
                    features_table.add_row("Thumbnails", f"✓ ({video.get('thumbnail_count', 0)})" if video.get('thumbnails') else "✗")
                    features_table.add_row("Searchable", "✓" if video.get('embeddings') else "✗")
                    features_table.add_row("Transcript", "✓" if video.get('transcript') else "✗")
                    
                    console.print(features_table)
                
                await monitor.watch_video(video_id, progress_callback)
            else:
                # One-time progress check
                progress = await monitor.get_video_progress(video_id)
                console.print(progress)
        else:
            # Show all active videos
            while True:
                console.clear()
                active = await monitor.get_active_videos()
                
                if not active:
                    console.print("[yellow]No videos currently processing[/yellow]")
                else:
                    table = Table(title=f"Active Processing ({len(active)} videos)")
                    table.add_column("File", style="cyan", max_width=30)
                    table.add_column("Progress", style="green")
                    table.add_column("Step", style="yellow")
                    table.add_column("Status", style="magenta")
                    table.add_column("Time", style="blue")
                    
                    for video in active:
                        progress_bar = f"[{'█' * (video['processing_progress'] // 10)}{'░' * (10 - video['processing_progress'] // 10)}]"
                        time_ago = f"{video['seconds_since_update']:.0f}s ago"
                        
                        table.add_row(
                            video['file_name'],
                            f"{progress_bar} {video['processing_progress']}%",
                            video.get('current_step', 'N/A'),
                            video['status'],
                            time_ago
                        )
                    
                    console.print(table)
                
                if not follow:
                    break
                
                await asyncio.sleep(3)
    
    asyncio.run(run_monitor())

@app.command()
def list_steps():
    """List all available processing steps"""
    
    registry = StepRegistry()
    steps = registry.list_steps()
    
    for category, category_steps in sorted(steps.items()):
        console.print(f"\n[bold cyan]{category.upper()}[/bold cyan]")
        
        for step in category_steps:
            console.print(f"  [green]{step['name']}[/green] (v{step['version']})")
            console.print(f"    [dim]{step['description']}[/dim]")

# Helper functions
def find_video_files(path: Path, recursive: bool, pattern: str) -> List[Path]:
    """Find video files matching pattern"""
    patterns = pattern.split(',')
    files = []
    
    if path.is_file():
        return [path]
    
    for pattern in patterns:
        if recursive:
            files.extend(path.rglob(pattern.strip()))
        else:
            files.extend(path.glob(pattern.strip()))
    
    return sorted(set(files))

def format_file_size(bytes_size: int) -> str:
    """Format file size in bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"

async def watch_directory(path: Path, pipeline: Pipeline, pattern: str, recursive: bool, auth_manager: AuthManager):
    """Watch directory for new files"""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    
    class VideoHandler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory:
                # Check if it matches our pattern
                for p in pattern.split(','):
                    if event.src_path.endswith(p.strip().replace('*', '')):
                        asyncio.create_task(
                            pipeline.process_video(event.src_path, auth_manager)
                        )
                        console.print(f"[green]New file queued: {event.src_path}[/green]")
                        break
    
    handler = VideoHandler()
    observer = Observer()
    observer.schedule(handler, str(path), recursive=recursive)
    observer.start()
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    app()
```

### 13. Worker Implementation (`worker.py`) - UPDATED FOR PROCRASTINATE V2.x

```python
from typing import Dict, List, Optional
import structlog
import asyncio

logger = structlog.get_logger()

async def run_workers(
    app,  # Procrastinate app
    worker_count: int,
    concurrency_config: Dict[str, int],
    queues: Optional[List[str]] = None
):
    """UPDATED: Run Procrastinate workers with per-queue concurrency using current patterns"""
    
    # Build concurrency map
    if queues:
        # Filter to requested queues
        concurrency = {q: concurrency_config.get(q, 1) for q in queues}
    else:
        concurrency = concurrency_config
    
    logger.info(
        "Starting workers with updated patterns",
        worker_count=worker_count,
        concurrency=concurrency
    )
    
    # UPDATED: Use current worker patterns with proper error handling
    try:
        await app.run_worker_async(
            concurrency=sum(concurrency.values()),  # Total concurrency
            queues=list(concurrency.keys()),
            wait=True,
            install_signal_handlers=True,  # Enable graceful shutdown
            listen_notify=True  # Enable real-time job notifications
        )
    except asyncio.CancelledError:
        logger.info("Worker cancelled, shutting down gracefully")
        raise
    except Exception as e:
        logger.error(f"Worker error: {str(e)}", exc_info=True)
        raise

# UPDATED: Add worker health monitoring
async def run_worker_with_monitoring(
    app,
    worker_count: int, 
    concurrency_config: Dict[str, int],
    queues: Optional[List[str]] = None,
    shutdown_timeout: int = 30
):
    """Run workers with health monitoring and graceful shutdown"""
    
    logger.info("Starting monitored worker process")
    
    try:
        # UPDATED: Use shutdown_graceful_timeout for better process management
        await app.run_worker_async(
            concurrency=sum(concurrency_config.values()),
            queues=queues or list(concurrency_config.keys()),
            wait=True,
            shutdown_graceful_timeout=shutdown_timeout,
            install_signal_handlers=True
        )
    except asyncio.CancelledError:
        logger.info("Graceful shutdown completed")
    except Exception as e:
        logger.error(f"Worker process failed: {str(e)}", exc_info=True)
        raise
```

### 14. Example Pipeline Configurations

#### Default Pipeline (`configs/default.yaml`)

```yaml
name: "default"
description: "Standard processing pipeline with incremental saves"
version: "1.0"

global_settings:
  supabase_url: "${SUPABASE_URL}"
  supabase_key: "${SUPABASE_ANON_KEY}"
  save_immediately: true  # Each step saves its results

steps:
  - category: checksum
    name: md5_checksum
    config:
      enabled: true
      queue: metadata
      priority: 10
      retry: 2

  - category: metadata
    name: ffmpeg_extractor
    config:
      enabled: true
      queue: metadata
      priority: 20
      retry: 3
      save_partial: true  # Save basic info first, then detailed
      params:
        extract_streams: true

  - category: thumbnails
    name: parallel_thumbs
    config:
      enabled: true
      queue: thumbnails
      priority: 30
      params:
        count: 5
        width: 1920
        quality: 90

  - category: compression
    name: ffmpeg_compress
    config:
      enabled: true
      queue: compression
      priority: 40
      params:
        codec: "hevc_videotoolbox"
        bitrate: "1000k"
        fps: 5

  - category: ai_analysis
    name: gemini_streaming
    config:
      enabled: false  # Enable with --enable-ai flag
      queue: ai_analysis
      priority: 50
      timeout: 600
      save_partial: true  # Stream results as they come
      params:
        api_key: "${GEMINI_API_KEY}"
        model: "gemini-2.5-flash"

  - category: embeddings
    name: bge_embeddings
    config:
      enabled: false
      queue: embeddings
      priority: 60
      params:
        api_key: "${DEEPINFRA_API_KEY}"

worker_config:
  metadata: 8
  thumbnails: 4
  compression: 2
  ai_analysis: 1
  embeddings: 4
```

#### Fast Pipeline (`configs/fast.yaml`)

```yaml
name: "fast"
description: "Quick processing - parallel steps, no AI"
version: "1.0"

global_settings:
  supabase_url: "${SUPABASE_URL}"
  supabase_key: "${SUPABASE_ANON_KEY}"

steps:
  - category: checksum
    name: blake3_checksum  # Faster than MD5
    config:
      enabled: true
      queue: metadata

  # These two can run in parallel!
  - category: metadata
    name: ffmpeg_extractor
    config:
      enabled: true
      queue: metadata
      params:
        basic_only: true  # Just duration, resolution

  - category: thumbnails
    name: ffmpeg_thumbs  # Faster than OpenCV
    config:
      enabled: true
      queue: thumbnails
      params:
        count: 3
        width: 1280
      parallel: true  # Can run while metadata runs

worker_config:
  metadata: 16  # High parallelism
  thumbnails: 8
```

#### AI Research Pipeline (`configs/ai_research.yaml`)

```yaml
name: "ai_research"
description: "Compare different AI models on same videos"
version: "1.0"

steps:
  - category: checksum
    name: md5_checksum
    config:
      enabled: true

  - category: metadata
    name: ffmpeg_extractor
    config:
      enabled: true

  - category: compression
    name: ffmpeg_compress
    config:
      enabled: true
      params:
        fps: 10  # Higher FPS for better AI analysis
        bitrate: "2000k"

  # Run multiple AI analyzers
  - category: ai_analysis
    name: gemini_streaming
    config:
      enabled: true
      queue: ai_gemini
      params:
        model: "gemini-2.5-flash"
        save_to_table: "ai_results_gemini"

  - category: ai_analysis
    name: claude_analyzer
    config:
      enabled: true
      queue: ai_claude
      params:
        model: "claude-3-opus"
        save_to_table: "ai_results_claude"

  - category: ai_analysis
    name: local_llava
    config:
      enabled: true
      queue: ai_local
      params:
        model: "llava-v1.6"
        save_to_table: "ai_results_local"

worker_config:
  metadata: 4
  compression: 2
  ai_gemini: 1
  ai_claude: 1
  ai_local: 1  # Each AI gets its own queue
```

### 15. Environment Variables

```bash
# .env file
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# For Procrastinate (PostgreSQL direct connection)
SUPABASE_DB_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres

# AI Services
GEMINI_API_KEY=your-gemini-key
DEEPINFRA_API_KEY=your-deepinfra-key

# OpenAI (for embeddings fallback)
OPENAI_API_KEY=your-openai-key
```

## Usage Examples

### Basic Usage

```bash
# Authentication
video-tool auth signup
video-tool auth login
video-tool auth status

# Process videos with default pipeline
video-tool ingest /path/to/videos

# Use specific configuration
video-tool ingest /videos --config fast

# Process with workers
video-tool ingest /videos --workers 8

# Run workers separately
video-tool worker --workers 4 --queue metadata,thumbnails

# Watch directory for new files
video-tool ingest /videos --watch --workers 4

# Check processing status
video-tool status --watch

# Monitor specific video
video-tool monitor --video abc123 --follow
```

### Search Examples

```bash
# Hybrid search (default - combines full-text and semantic)
video-tool search query "cats playing outdoors"

# Semantic search only
video-tool search query "birthday party" --type semantic

# Full-text search
video-tool search query "mountain hiking" --type fulltext

# Transcript search
video-tool search query "hello everyone" --type transcripts

# Find similar videos
video-tool search similar abc123def456 --limit 10

# Show detailed clip info
video-tool search info abc123def456 --transcript --analysis

# Show catalog statistics
video-tool search stats
```

### Advanced Usage

```bash
# AI-enabled processing
video-tool ingest /videos --config ai_research --workers 4

# Process with custom settings
video-tool ingest /videos \
  --config default \
  --workers 8 \
  --pattern "*.mp4,*.mov,*.mkv" \
  --recursive

# Monitor processing
video-tool monitor --follow  # All videos
video-tool monitor --video abc123 --follow  # Specific video

# Check system status
video-tool status
video-tool list-steps
```

## Key Benefits

1. **Modular Architecture**: Each step is independent and swappable
2. **Incremental Saves**: Immediate visibility and fault tolerance  
3. **Authentication & RLS**: Secure multi-user system
4. **Comprehensive Search**: Semantic, full-text, and hybrid search
5. **Real-time Monitoring**: Track progress with live updates
6. **Parallel Processing**: Configurable concurrency per step type
7. **Resume Capability**: Automatic recovery from failures
8. **Vector Embeddings**: Advanced semantic search capabilities
9. **Queue-based**: Scalable with Procrastinate task queuing
10. **User Isolation**: Each user sees only their own data

## 🚀 PROCRASTINATE V2.x UPDATES & STREAMLINING

### Key Changes Made for Current Best Practices

#### 1. **Database Connection Patterns**
- **BEFORE**: Manual parameter parsing and connection setup
- **AFTER**: Direct `conninfo` string usage with `PsycopgConnector(conninfo=url)`
- **BENEFIT**: Simpler, more reliable connection management

#### 2. **Schema Initialization**
- **BEFORE**: Manual SQL execution with `procrastinate.create_all()`
- **AFTER**: Use `procrastinate_schema.create_all()` with proper async patterns
- **BENEFIT**: Aligned with current Procrastinate schema management

#### 3. **Task Registration**
- **BEFORE**: Simple `@app.task()` decorators
- **AFTER**: Enhanced with `pass_context=True` for better error handling and monitoring
- **BENEFIT**: Access to job context, better debugging, enhanced retry logic

#### 4. **Worker Management**
- **BEFORE**: Basic `app.run_worker()` calls
- **AFTER**: Added `shutdown_graceful_timeout`, `install_signal_handlers`, `listen_notify`
- **BENEFIT**: Graceful shutdown, real-time job notifications, better process management

#### 5. **Error Handling**
- **BEFORE**: Simple exception raising in steps
- **AFTER**: Return failed `StepResult` objects, let Procrastinate handle retries
- **BENEFIT**: Better retry logic, more predictable error handling

#### 6. **Connection Management**
- **BEFORE**: Global database instance
- **AFTER**: Proper singleton pattern with cleanup methods
- **BENEFIT**: Better resource management, cleaner shutdown

### Streamlining Opportunities

#### **Option A: Keep Full Architecture (Recommended)**
- **Pros**: Maximum flexibility, comprehensive feature set, production-ready
- **Cons**: More complex setup, larger codebase
- **Use Case**: Production systems, complex video processing workflows

#### **Option B: Simplified Architecture**
If you want to streamline further, here's what could be simplified:

```python
# Simplified Step Base Class
class SimpleStep(ABC):
    name: str = ""
    
    @abstractmethod
    async def process(self, video_data: dict) -> dict:
        """Just return the data to merge"""
        pass

# Simplified Task Registration
@app.task(queue="processing", pass_context=True)
async def process_video_step(context, video_id: str, step_name: str):
    # Load step class dynamically
    step = get_step_class(step_name)
    
    # Get video data from database
    video_data = await get_video_data(video_id)
    
    # Process
    result = await step.process(video_data)
    
    # Save results
    await save_video_data(video_id, result)
    
    return result
```

#### **Recommended Approach: Hybrid Simplification**

Keep the comprehensive architecture but add a simplified API for basic use cases:

```python
# Simple API for basic processing
@app.command()
def quick_process(file_path: str):
    """Quick processing with default pipeline"""
    asyncio.run(simple_video_process(file_path))

async def simple_video_process(file_path: str):
    """Simplified processing for basic use cases"""
    # Auto-detect user, queue basic steps
    steps = ["checksum", "metadata", "thumbnails"]
    
    for step in steps:
        await app.task(f"simple.{step}").defer_async(file_path=file_path)
```

### Performance Optimizations

#### 1. **Parallel Step Execution**
```yaml
# In pipeline config - steps that can run in parallel
parallel_groups:
  - name: "metadata_group"
    steps: ["checksum", "ffmpeg_metadata", "exif_metadata"]
    wait_for_all: true
  
  - name: "analysis_group" 
    steps: ["thumbnails", "exposure_analysis"]
    depends_on: ["metadata_group"]
```

#### 2. **Smart Queue Management**
```python
# Auto-scale workers based on queue depth
async def smart_worker_scaling():
    job_counts = await app.job_manager.count_by_status()
    
    if job_counts.get('todo', 0) > 100:
        # Scale up workers
        await start_additional_workers()
    elif job_counts.get('todo', 0) < 10:
        # Scale down workers  
        await stop_excess_workers()
```

#### 3. **Caching Layer**
```python
# Add Redis caching for metadata
@lru_cache(maxsize=1000)
async def get_cached_metadata(file_checksum: str):
    # Check cache first, then database
    pass
```

### Migration Path

#### **Phase 1: Update Core (1-2 days)**
1. ✅ Update database connection patterns
2. ✅ Update task registration with `pass_context=True`
3. ✅ Update worker management with graceful shutdown
4. ✅ Test basic video processing pipeline

#### **Phase 2: Enhanced Features (3-5 days)**
1. Add parallel step execution
2. Implement smart queue management
3. Add comprehensive monitoring dashboard
4. Performance optimization and caching

#### **Phase 3: Production Deployment (2-3 days)**
1. Add proper logging and metrics
2. Container deployment with Docker
3. CI/CD pipeline setup
4. Load testing and optimization

### Current Architecture Benefits

✅ **Modular**: Each step is independent and swappable
✅ **Scalable**: Queue-based processing with configurable concurrency  
✅ **Fault-Tolerant**: Automatic retry, resume capability, incremental saves
✅ **User-Isolated**: RLS policies ensure data security
✅ **Searchable**: Comprehensive semantic and full-text search
✅ **Monitorable**: Real-time progress tracking and health checks
✅ **Production-Ready**: Authentication, logging, error handling

This architecture strikes the right balance between functionality and maintainability while following current Procrastinate best practices.

#### HDR Metadata Extractor (`steps/metadata/hdr_extractor.py`)

```python
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
        "is_hdr", "hdr_format", "hdr_format_commercial", "transfer_characteristics",
        "color_space", "color_primaries", "matrix_coefficients", "color_range",
        "master_display", "max_cll", "max_fall", "bit_depth_video"
    ]
    
    def __init__(self, config):
        super().__init__(config) 
        self.pymediainfo_available = self._check_pymediainfo()
        
    def _check_pymediainfo(self) -> bool:
        """Check if PyMediaInfo is available"""
        try:
            import pymediainfo
            return True
        except ImportError:
            self.logger.warning("PyMediaInfo not available for HDR metadata extraction")
            self.logger.info("Install: pip install pymediainfo")
            return False
    
    async def process(self, video: VideoMetadata) -> StepResult:
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
            combined_metadata = {**hdr_metadata, **codec_metadata}
            
            # Log important findings
            if combined_metadata.get("is_hdr"):
                hdr_format = combined_metadata.get("hdr_format", "Unknown")
                self.logger.info(f"HDR video detected: {hdr_format}")
            
            return StepResult(
                success=True,
                step_name=self.name,
                video_id=video.video_id,
                data=combined_metadata,
                metadata={
                    "extraction_method": "pymediainfo",
                    "hdr_detected": combined_metadata.get("is_hdr", False),
                    "bit_depth": combined_metadata.get("bit_depth_video")
                }
            )
            
        except Exception as e:
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
        import pymediainfo
        
        try:
            # Parse media file
            media_info = pymediainfo.MediaInfo.parse(file_path)
            
            # Find the first video track
            video_track = next((track for track in media_info.tracks if track.track_type == 'Video'), None)
            
            if not video_track:
                self.logger.warning("No video track found in file")
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
            self.logger.error(f"Failed to extract HDR metadata: {str(e)}")
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
        import pymediainfo
        
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
                
            if hasattr(video_track, 'scan_order') and video_track.scan_order:
                codec_params["field_order"] = str(video_track.scan_order)
            
            return codec_params
            
        except Exception as e:
            self.logger.error(f"Failed to extract codec parameters: {str(e)}")
            return {}
    
    async def setup(self):
        """Setup method called once when step is initialized"""
        if self.pymediainfo_available:
            self.logger.info("HDR Metadata Extractor step initialized")
            self.logger.info("Supported HDR formats: HDR10, HDR10+, Dolby Vision, HLG")
        else:
            self.logger.info("HDR Metadata Extractor step disabled (PyMediaInfo unavailable)")
    
    def get_info(self) -> Dict[str, Any]:
        """Get step information"""
        info = super().get_info()
        info.update({
            "pymediainfo_available": self.pymediainfo_available,
            "supported_hdr_formats": ["HDR10", "HDR10+", "Dolby Vision", "HLG", "DCI-P3"],
            "color_metadata": [
                "color_space", "color_primaries", "matrix_coefficients", 
                "color_range", "transfer_characteristics"
            ],
            "hdr_metadata": [
                "master_display", "max_cll", "max_fall", "bit_depth"
            ]
        })
        return info
```



#### PyExifTool Metadata Extractor (`steps/metadata/exiftool_extractor.py`)

```python
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
        processed = {}
        
        # Camera information
        processed["camera_make"] = raw_metadata.get('EXIF:Make') or raw_metadata.get('QuickTime:Make')
        processed["camera_model"] = raw_metadata.get('EXIF:Model') or raw_metadata.get('QuickTime:Model')
        processed["lens_model"] = raw_metadata.get('EXIF:LensModel') or raw_metadata.get('QuickTime:LensModel')
        processed["camera_serial_number"] = raw_metadata.get('EXIF:SerialNumber') or raw_metadata.get('QuickTime:SerialNumber')
        
        # Software information
        processed["software"] = raw_metadata.get('EXIF:Software') or raw_metadata.get('QuickTime:Software')
        
        # Focal length processing
        focal_length_raw = raw_metadata.get('EXIF:FocalLength') or raw_metadata.get('QuickTime:FocalLength')
        if focal_length_raw:
            focal_length_mm = self._parse_focal_length(focal_length_raw)
            if focal_length_mm:
                processed["focal_length_mm"] = focal_length_mm
                processed["focal_length_category"] = self._categorize_focal_length(focal_length_mm)
                processed["focal_length_source"] = "EXIF"
        
        # Camera settings
        processed["iso"] = self._parse_numeric(raw_metadata.get('EXIF:ISO') or raw_metadata.get('QuickTime:ISO'))
        processed["shutter_speed"] = self._parse_shutter_speed(
            raw_metadata.get('EXIF:ShutterSpeedValue') or 
            raw_metadata.get('EXIF:ExposureTime') or 
            raw_metadata.get('QuickTime:ExposureTime')
        )
        processed["f_stop"] = self._parse_numeric(
            raw_metadata.get('EXIF:FNumber') or 
            raw_metadata.get('EXIF:ApertureValue') or 
            raw_metadata.get('QuickTime:Aperture')
        )
        
        # Exposure and white balance
        processed["exposure_mode"] = self._map_exposure_mode(raw_metadata.get('EXIF:ExposureMode'))
        processed["white_balance"] = self._map_white_balance(raw_metadata.get('EXIF:WhiteBalance'))
        
        # GPS information
        gps_lat = raw_metadata.get('EXIF:GPSLatitude') or raw_metadata.get('QuickTime:GPSLatitude')
        gps_lon = raw_metadata.get('EXIF:GPSLongitude') or raw_metadata.get('QuickTime:GPSLongitude') 
        gps_alt = raw_metadata.get('EXIF:GPSAltitude') or raw_metadata.get('QuickTime:GPSAltitude')
        
        if gps_lat:
            processed["gps_latitude"] = self._parse_gps_coordinate(gps_lat)
        if gps_lon:
            processed["gps_longitude"] = self._parse_gps_coordinate(gps_lon)
        if gps_alt:
            processed["gps_altitude"] = self._parse_numeric(gps_alt)
        
        # Location information
        processed["location_name"] = self._build_location_name(raw_metadata)
        
        # Timestamps
        processed["date_taken"] = self._parse_datetime(
            raw_metadata.get('EXIF:DateTimeOriginal') or 
            raw_metadata.get('EXIF:CreateDate') or 
            raw_metadata.get('QuickTime:CreateDate')
        )
        processed["date_created"] = self._parse_datetime(raw_metadata.get('File:FileCreateDate'))
        processed["date_modified"] = self._parse_datetime(raw_metadata.get('File:FileModifyDate'))
        
        # Additional metadata
        processed["artist"] = raw_metadata.get('EXIF:Artist') or raw_metadata.get('QuickTime:Artist')
        processed["copyright"] = raw_metadata.get('EXIF:Copyright') or raw_metadata.get('QuickTime:Copyright')
        processed["description"] = raw_metadata.get('EXIF:ImageDescription') or raw_metadata.get('QuickTime:Description')
        
        # Filter out None values
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
```



## Enhanced Data Models (`core/enhanced_models.py`)

Your original implementation has very detailed Pydantic models that provide comprehensive structure for video metadata. Here are the enhanced models that should be added to complement the basic models:

```python
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

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
    hdr: VideoHDRDetails = VideoHDRDetails()

class VideoExposureDetails(BaseModel):
    """Exposure analysis results"""
    warning: Optional[bool] = None
    stops: Optional[float] = None  # Exposure deviation in stops
    overexposed_percentage: Optional[float] = None
    underexposed_percentage: Optional[float] = None
    overall_quality: Optional[str] = None  # excellent, good, fair, poor

class VideoDetails(BaseModel):
    """Complete video technical metadata"""
    codec: VideoCodecDetails = VideoCodecDetails()
    container: Optional[str] = None
    resolution: VideoResolution = VideoResolution()
    frame_rate: Optional[float] = None
    color: VideoColorDetails = VideoColorDetails()
    exposure: VideoExposureDetails = VideoExposureDetails()

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
    focal_length: CameraFocalLength = CameraFocalLength()
    settings: CameraSettings = CameraSettings()
    location: CameraLocation = CameraLocation()

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
    params: Dict[str, Any] = {}
    dependencies: List[str] = []
    
    # Advanced configuration
    save_immediately: bool = True
    save_partial: bool = False
    requires_auth: bool = True
    output_artifacts: List[str] = []

class PipelineConfiguration(BaseModel):
    """Enhanced pipeline configuration"""
    name: str
    description: str = ""
    version: str = "1.0"
    
    # Steps configuration
    steps: List[StepConfiguration] = []
    
    # Global settings
    global_settings: Dict[str, Any] = {}
    
    # Worker configuration
    worker_config: Dict[str, int] = {
        "default": 4,
        "metadata": 8,
        "thumbnails": 4,
        "compression": 2,
        "ai_analysis": 1,
        "embeddings": 4
    }
    
    # Processing settings
    enable_incremental_saves: bool = True
    enable_progress_tracking: bool = True
    enable_error_recovery: bool = True
    
    # Output settings
    create_individual_files: bool = True
    organize_by_timestamp: bool = True
    generate_summary_files: bool = True

import uuid
```

These enhanced models provide:

1. **Detailed Technical Metadata**: Comprehensive codec, resolution, HDR, and color information
2. **Complete Camera Information**: EXIF data, focal length categorization, GPS location
3. **Rich AI Analysis Models**: Shot types, quality assessment, content analysis
4. **Structured Audio Analysis**: Transcripts, speaker detection, sound events
5. **Content Understanding**: Entity detection, activity recognition, content warnings
6. **Enhanced Configuration**: Advanced pipeline and step configuration options

The models maintain backward compatibility while providing much richer structure for your video processing results.



## Output Organization & Run Management (`core/output_manager.py`)

Your original implementation has sophisticated output organization with timestamped run directories. Here's how to integrate this into the modular architecture:

```python
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import structlog

from video_tool.core.models import EnhancedVideoIngestOutput

logger = structlog.get_logger(__name__)

class OutputManager:
    """
    Manages organized output structure with timestamped runs,
    similar to your original implementation but integrated with the modular system.
    """
    
    def __init__(self, base_output_dir: Optional[str] = None):
        # Determine base output directory
        if base_output_dir:
            self.base_output_dir = Path(base_output_dir)
        else:
            # Default to output directory in project root (like original)
            package_dir = Path(__file__).parent.parent
            self.base_output_dir = package_dir / "output"
        
        self.current_run_dir = None
        self.current_run_id = None
        
        # Setup directory structure
        self.setup_base_directories()
    
    def setup_base_directories(self):
        """Setup the base directory structure"""
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Main directories (like your original structure)
        self.runs_dir = self.base_output_dir / "runs"
        self.global_json_dir = self.base_output_dir / "json"
        self.global_logs_dir = self.base_output_dir / "logs"
        
        # Create directories
        for directory in [self.runs_dir, self.global_json_dir, self.global_logs_dir]:
            directory.mkdir(exist_ok=True)
        
        logger.info(f"Output directories initialized at {self.base_output_dir}")
    
    def create_new_run(self, run_name: Optional[str] = None) -> str:
        """
        Create a new timestamped run directory.
        
        Args:
            run_name: Optional custom name, otherwise uses timestamp
            
        Returns:
            Run ID string 
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if run_name:
            run_id = f"{run_name}_{timestamp}"
        else:
            run_id = f"run_{timestamp}"
        
        self.current_run_id = run_id
        self.current_run_dir = self.runs_dir / run_id
        
        # Create run directory structure
        self._create_run_directories()
        
        # Create run manifest
        self._create_run_manifest()
        
        logger.info(f"Created new run: {run_id}")
        return run_id
    
    def _create_run_directories(self):
        """Create the directory structure for a run"""
        if not self.current_run_dir:
            raise ValueError("No current run directory set")
        
        # Core directories (matching your original structure)
        directories = [
            "json",           # Individual video JSON files
            "thumbnails",     # Video thumbnails organized by checksum
            "ai_analysis",    # Detailed AI analysis files
            "compressed",     # Compressed videos for AI processing
            "logs",          # Run-specific logs
            "artifacts",     # Step artifacts and intermediate files
            "reports",       # Summary reports and statistics
            "exports"        # Export-ready files
        ]
        
        for directory in directories:
            (self.current_run_dir / directory).mkdir(parents=True, exist_ok=True)
        
        # Store directory paths for easy access
        self.run_paths = {
            "base": self.current_run_dir,
            "json": self.current_run_dir / "json",
            "thumbnails": self.current_run_dir / "thumbnails", 
            "ai_analysis": self.current_run_dir / "ai_analysis",
            "compressed": self.current_run_dir / "compressed",
            "logs": self.current_run_dir / "logs",
            "artifacts": self.current_run_dir / "artifacts",
            "reports": self.current_run_dir / "reports",
            "exports": self.current_run_dir / "exports"
        }
    
    def _create_run_manifest(self):
        """Create manifest file for the run"""
        manifest = {
            "run_id": self.current_run_id,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "videos_processed": 0,
            "videos_failed": 0,
            "total_duration_seconds": 0.0,
            "total_size_bytes": 0,
            "pipeline_config": {},
            "processing_stats": {
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "total_processing_time": None
            }
        }
        
        manifest_path = self.current_run_dir / "run_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def save_video_output(self, video_output: EnhancedVideoIngestOutput) -> Dict[str, str]:
        """
        Save complete video output in organized structure.
        
        Args:
            video_output: Complete video processing results
            
        Returns:
            Dict of saved file paths
        """
        if not self.current_run_dir:
            raise ValueError("No active run. Call create_new_run() first.")
        
        saved_paths = {}
        
        # 1. Save individual video JSON
        json_filename = f"{video_output.file_info.file_name}_{video_output.id}.json"
        json_path = self.run_paths["json"] / json_filename
        
        with open(json_path, 'w') as f:
            json.dump(video_output.dict(), f, indent=2, default=str)
        saved_paths["individual_json"] = str(json_path)
        
        # 2. Save to global JSON directory (like original)
        global_json_path = self.global_json_dir / json_filename
        shutil.copy2(json_path, global_json_path)
        saved_paths["global_json"] = str(global_json_path)
        
        # 3. Organize thumbnails by checksum (like original)
        if video_output.thumbnails:
            checksum = video_output.file_info.file_checksum
            thumbnail_dir = self.run_paths["thumbnails"] / f"{checksum}"
            thumbnail_dir.mkdir(exist_ok=True)
            
            organized_thumbnails = []
            for i, thumbnail_path in enumerate(video_output.thumbnails):
                if Path(thumbnail_path).exists():
                    # Create organized filename
                    thumb_filename = f"thumb_{i:03d}_{checksum}.jpg"
                    organized_path = thumbnail_dir / thumb_filename
                    
                    # Copy to organized location
                    shutil.copy2(thumbnail_path, organized_path)
                    organized_thumbnails.append(str(organized_path))
            
            saved_paths["thumbnails"] = organized_thumbnails
        
        # 4. Save detailed AI analysis if available
        if (video_output.analysis.ai_analysis and 
            video_output.analysis.ai_analysis.analysis_file_path):
            
            ai_analysis_name = f"{video_output.file_info.file_name}_AI_analysis.json"
            ai_analysis_path = self.run_paths["ai_analysis"] / ai_analysis_name
            
            # Copy the original AI analysis file
            original_path = video_output.analysis.ai_analysis.analysis_file_path
            if Path(original_path).exists():
                shutil.copy2(original_path, ai_analysis_path)
                saved_paths["ai_analysis"] = str(ai_analysis_path)
        
        # 5. Update run statistics
        self._update_run_stats(video_output)
        
        logger.info(f"Saved organized output for {video_output.file_info.file_name}")
        return saved_paths
    
    def save_run_summary(self, videos: List[EnhancedVideoIngestOutput], 
                        processing_stats: Dict[str, Any]) -> str:
        """
        Save comprehensive run summary (like your original all_videos_*.json).
        
        Args:
            videos: List of all processed videos
            processing_stats: Overall processing statistics
            
        Returns:
            Path to summary file
        """
        if not self.current_run_dir:
            raise ValueError("No active run.")
        
        # Create comprehensive summary
        summary = {
            "run_info": {
                "run_id": self.current_run_id,
                "created_at": datetime.now().isoformat(),
                "total_videos": len(videos),
                "successful_videos": len([v for v in videos if v.processing_status == "completed"]),
                "failed_videos": len([v for v in videos if v.processing_status == "failed"])
            },
            "processing_stats": processing_stats,
            "videos": [video.dict() for video in videos]
        }
        
        # Save to run directory
        summary_filename = f"all_videos_{self.current_run_id}.json"
        summary_path = self.run_paths["reports"] / summary_filename
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Also save to global directory
        global_summary_path = self.global_json_dir / summary_filename
        shutil.copy2(summary_path, global_summary_path)
        
        logger.info(f"Saved run summary: {summary_filename}")
        return str(summary_path)
    
    def save_pipeline_config(self, config: Dict[str, Any]) -> str:
        """Save pipeline configuration for the run"""
        if not self.current_run_dir:
            raise ValueError("No active run.")
        
        config_path = self.current_run_dir / "pipeline_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return str(config_path)
    
    def generate_processing_report(self) -> str:
        """Generate comprehensive processing report"""
        if not self.current_run_dir:
            raise ValueError("No active run.")
        
        # Read manifest for stats
        manifest_path = self.current_run_dir / "run_manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Collect individual video results
        json_files = list(self.run_paths["json"].glob("*.json"))
        video_results = []
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                video_data = json.load(f)
                video_results.append(video_data)
        
        # Generate report
        report = {
            "run_summary": manifest,
            "video_statistics": self._generate_video_statistics(video_results),
            "processing_timeline": self._generate_processing_timeline(video_results),
            "quality_analysis": self._generate_quality_analysis(video_results),
            "technical_summary": self._generate_technical_summary(video_results)
        }
        
        # Save report
        report_filename = f"processing_report_{self.current_run_id}.json"
        report_path = self.run_paths["reports"] / report_filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(report_path)
    
    def _update_run_stats(self, video_output: EnhancedVideoIngestOutput):
        """Update run statistics"""
        manifest_path = self.current_run_dir / "run_manifest.json"
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Update counts
        manifest["videos_processed"] += 1
        if video_output.analysis:
            manifest["total_duration_seconds"] += float(video_output.video.frame_rate or 0)
        manifest["total_size_bytes"] += video_output.file_info.file_size_bytes
        
        # Update status
        manifest["last_updated"] = datetime.now().isoformat()
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _generate_video_statistics(self, video_results: List[Dict]) -> Dict[str, Any]:
        """Generate video statistics for report"""
        if not video_results:
            return {}
        
        stats = {
            "total_videos": len(video_results),
            "total_duration_hours": sum(
                v.get("video", {}).get("duration_seconds", 0) for v in video_results
            ) / 3600,
            "total_size_gb": sum(
                v.get("file_info", {}).get("file_size_bytes", 0) for v in video_results  
            ) / (1024**3),
            "camera_distribution": {},
            "resolution_distribution": {},
            "codec_distribution": {}
        }
        
        # Analyze distributions
        for video in video_results:
            camera_info = video.get("camera", {})
            if camera_info.get("make"):
                camera = f"{camera_info['make']} {camera_info.get('model', '')}"
                stats["camera_distribution"][camera] = stats["camera_distribution"].get(camera, 0) + 1
            
            video_info = video.get("video", {})
            resolution = video_info.get("resolution", {})
            if resolution.get("width") and resolution.get("height"):
                res_key = f"{resolution['width']}x{resolution['height']}"
                stats["resolution_distribution"][res_key] = stats["resolution_distribution"].get(res_key, 0) + 1
            
            codec = video_info.get("codec", {}).get("name")
            if codec:
                stats["codec_distribution"][codec] = stats["codec_distribution"].get(codec, 0) + 1
        
        return stats
    
    def _generate_processing_timeline(self, video_results: List[Dict]) -> List[Dict[str, Any]]:
        """Generate processing timeline"""
        timeline = []
        
        for video in video_results:
            file_info = video.get("file_info", {})
            timeline.append({
                "file_name": file_info.get("file_name"),
                "processed_at": file_info.get("processed_at"),
                "processing_status": video.get("processing_status"),
                "duration_seconds": video.get("video", {}).get("duration_seconds")
            })
        
        # Sort by processing time
        timeline.sort(key=lambda x: x.get("processed_at", ""))
        return timeline
    
    def _generate_quality_analysis(self, video_results: List[Dict]) -> Dict[str, Any]:
        """Generate quality analysis summary"""
        quality_stats = {             
            "hdr_videos": 0,
            "4k_videos": 0,
            "exposure_warnings": 0,
            "average_technical_quality": {}
        }
        
        for video in video_results:
            # HDR detection
            color_info = video.get("video", {}).get("color", {}).get("hdr", {})
            if color_info.get("is_hdr"):
                quality_stats["hdr_videos"] += 1
            
            # 4K detection
            resolution = video.get("video", {}).get("resolution", {})
            if resolution.get("height", 0) >= 2160:
                quality_stats["4k_videos"] += 1
            
            # Exposure warnings
            exposure = video.get("video", {}).get("exposure", {})
            if exposure.get("warning"):
                quality_stats["exposure_warnings"] += 1
        
        return quality_stats
    
    def _generate_technical_summary(self, video_results: List[Dict]) -> Dict[str, Any]:
        """Generate technical summary"""
        return {
            "total_files_processed": len(video_results),
            "average_file_size_mb": sum(
                v.get("file_info", {}).get("file_size_bytes", 0) for v in video_results
            ) / len(video_results) / (1024**2) if video_results else 0,
            "processing_completion_rate": len([
                v for v in video_results if v.get("processing_status") == "completed"
            ]) / len(video_results) if video_results else 0
        }
    
    def get_run_info(self) -> Optional[Dict[str, Any]]:
        """Get current run information"""
        if not self.current_run_dir:
            return None
        
        manifest_path = self.current_run_dir / "run_manifest.json"
        if not manifest_path.exists():
            return None
        
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    def list_available_runs(self) -> List[Dict[str, Any]]:
        """List all available runs"""
        runs = []
        
        for run_dir in self.runs_dir.iterdir():
            if run_dir.is_dir():
                manifest_path = run_dir / "run_manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        runs.append({
                            "run_id": manifest.get("run_id"),
                            "created_at": manifest.get("created_at"),
                            "status": manifest.get("status"),
                            "videos_processed": manifest.get("videos_processed", 0),
                            "path": str(run_dir)
                        })
        
        # Sort by creation time (newest first)
        runs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return runs
    
    def cleanup_old_runs(self, keep_count: int = 10):
        """Clean up old runs, keeping only the most recent ones"""
        runs = self.list_available_runs()
        
        if len(runs) <= keep_count:
            return
        
        # Remove oldest runs
        runs_to_remove = runs[keep_count:]
        
        for run_info in runs_to_remove:
            run_path = Path(run_info["path"])
            if run_path.exists():
                shutil.rmtree(run_path)
                logger.info(f"Cleaned up old run: {run_info['run_id']}")

# Global output manager instance
output_manager = OutputManager()

def get_output_manager() -> OutputManager:
    """Get the global output manager instance"""
    return output_manager
```

This Output Manager provides:

1. **Timestamped Run Directories**: Organized like your original with `run_YYYYMMDD_HHMMSS/`
2. **Structured Output Organization**: JSON, thumbnails, AI analysis, logs, etc.
3. **Global + Run-Specific Storage**: Files saved both in run directories and global locations
4. **Comprehensive Run Summaries**: Like your original `all_videos_*.json` files
5. **Processing Reports**: Detailed statistics and analysis
6. **Run Management**: List, cleanup, and manage multiple processing runs

This integrates seamlessly with the modular pipeline while maintaining your original sophisticated output organization.

