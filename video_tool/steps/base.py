from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
import structlog
import asyncio # Required for async methods

# Assuming models are in core.models
from video_tool.core.models import StepConfig, StepResult, VideoMetadata, ProcessingStatus
# Assuming db utilities are in core.db
from video_tool.core.db import get_supabase, get_db # get_db for initializing supabase if needed

logger = structlog.get_logger(__name__) # Changed to __name__ for better logging context

class BaseStep(ABC):
    """Base class for all processing steps with incremental saving and user context"""
    
    # Step metadata (must be overridden by subclasses)
    name: str = "" 
    version: str = "1.0"
    description: str = ""
    category: str = "" # e.g., "checksum", "metadata", "ai_analysis"
    
    # Dependencies
    requires: List[str] = []  # Required fields from VideoMetadata
    provides: List[str] = []  # Fields this step adds/updates in VideoMetadata
    optional_requires: List[str] = []  # Optional fields that enhance processing
    
    # Save configuration (can be overridden by step or pipeline config)
    save_immediately: bool = True  # Save results immediately after step completion
    save_partial: bool = False     # Save partial results during processing (for long-running steps)
    
    def __init__(self, config: StepConfig):
        self.config = config # Step-specific configuration
        # Bind step name to logger for contextual logging
        self.logger = logger.bind(step_name=self.name, step_category=self.category)
        self.supabase: Optional[Client] = None # Supabase client, initialized in execute
        self.task = None # To store procrastinate task reference, set by Pipeline

    async def _ensure_supabase_client(self):
        """Ensures the Supabase client is initialized."""
        if not self.supabase:
            # Ensure DB is initialized before getting Supabase client
            # This might be redundant if pipeline ensures DB is up.
            await get_db() 
            self.supabase = await get_supabase()
            if not self.supabase:
                self.logger.error("Supabase client could not be initialized for step.")
                raise RuntimeError("Supabase client not available for step execution.")

    # UPDATED: Enhanced error handling and context passing (as per plan)
    async def execute(self, video: VideoMetadata, step_index: int, total_steps: int, context: Optional[Dict[str, Any]] = None) -> StepResult:
        """
        Execute step with progress tracking, incremental saves, and user context.
        This method is called by the Procrastinate task.
        """
        await self._ensure_supabase_client()
        
        step_start_time = datetime.utcnow()
        self.logger.info("Executing step", video_id=video.video_id, step_idx=step_index, total_steps=total_steps)

        # Update progress: starting step
        await self._update_progress(video.video_id, video.user_id, 'starting', step_index, total_steps)
        
        try:
            # Validate inputs
            if not await self.validate_input(video):
                missing_fields = [field for field in self.requires if not hasattr(video, field) or getattr(video, field) is None]
                error_msg = f"Missing required inputs: {missing_fields}"
                self.logger.error(error_msg, video_id=video.video_id)
                # Update progress: failed step due to validation
                await self._update_progress(video.video_id, video.user_id, 'failed', step_index, total_steps, error_msg)
                return StepResult(
                    success=False, step_name=self.name, video_id=video.video_id,
                    error=error_msg, started_at=step_start_time
                )
            
            # UPDATED: Pass context to process method (as per plan)
            # The `context` here is the Procrastinate job context
            result = await self.process(video, context) 
            
            result.started_at = step_start_time # Ensure started_at is set on the result

            if result.success:
                # Save results immediately if configured
                if self.save_immediately and result.data: # Only save if there's data
                    await self._save_step_results(video.video_id, video.user_id, result)
                
                # Update progress: completed step
                await self._update_progress(video.video_id, video.user_id, 'completed', step_index, total_steps)
            else:
                # Update progress: failed step (failure reported by process method)
                self.logger.error("Step process method reported failure", video_id=video.video_id, error=result.error)
                await self._update_progress(video.video_id, video.user_id, 'failed', step_index, total_steps, result.error)
            
            return result # Return the StepResult from process()
            
        except Exception as e:
            self.logger.error(f"Step execution failed with unhandled exception: {str(e)}", video_id=video.video_id, exc_info=True)
            await self._update_progress(video.video_id, video.user_id, 'error', step_index, total_steps, str(e))
            
            # UPDATED: Return failed result instead of raising (as per plan)
            return StepResult(
                success=False, step_name=self.name, video_id=video.video_id,
                error=str(e), error_details={"exception_type": type(e).__name__},
                started_at=step_start_time
            )

    @abstractmethod
    async def process(self, video: VideoMetadata, context: Optional[Dict[str, Any]] = None) -> StepResult:
        """
        Process the video and return results. Must be implemented by subclasses.
        
        Args:
            video: Current video metadata (includes user_id if available from previous steps or initial creation)
            context: Optional additional context from Procrastinate job
            
        Returns:
            StepResult with data to be merged into video metadata
        """
        pass
    
    async def validate_input(self, video: VideoMetadata) -> bool:
        """Validate that required inputs are present in the VideoMetadata object."""
        video_dict = video.model_dump() # Use model_dump for Pydantic v2
        
        for field in self.requires:
            if field not in video_dict or video_dict[field] is None:
                self.logger.warning(f"Missing required field for step {self.name}: {field}", video_id=video.video_id)
                return False
        return True
    
    async def _update_progress(self, video_id: str, user_id: Optional[str], status: str, step_index: int, total_steps: int, error_msg: Optional[str] = None):
        """Update processing progress in database with user context."""
        await self._ensure_supabase_client()
        if not user_id:
            self.logger.warning("user_id not provided for progress update, skipping database update.", video_id=video_id)
            # Potentially log this to a local file or a different system if user_id is critical
            return

        progress_pct = int(((step_index) / total_steps) * 100) if total_steps > 0 else 0
        
        update_data_clips = { # Data for 'clips' table
            'current_step': self.name,
            'processing_progress': progress_pct,
            'total_steps': total_steps, # Store total steps for UI
            'updated_at': datetime.utcnow().isoformat()
        }
        
        # Update overall video status based on step status
        if status == 'completed' and step_index == total_steps: # Last step completed
            update_data_clips['status'] = ProcessingStatus.COMPLETED.value
            update_data_clips['last_step_completed'] = self.name
        elif status == 'failed' or status == 'error':
            update_data_clips['status'] = ProcessingStatus.FAILED.value
        elif status == 'starting' and update_data_clips.get('status') != ProcessingStatus.FAILED.value : # Don't override FAILED
             update_data_clips['status'] = ProcessingStatus.PROCESSING.value


        # Update processing_status JSONB field in 'clips' table
        # This field tracks individual step statuses
        try:
            result = await self.supabase.table('clips')\
                .select('processing_status')\
                .eq('id', video_id)\
                .eq('user_id', user_id)\
                .single()\
                .execute()
            
            existing_status_json = result.data.get('processing_status', {}) if result.data else {}
            if not isinstance(existing_status_json, dict): # Ensure it's a dict
                existing_status_json = {}

        except Exception as e:
            self.logger.error("Failed to fetch existing processing_status", video_id=video_id, error=e)
            existing_status_json = {}

        step_status_entry = {}
        if status == 'completed':
            step_status_entry = {'status': 'completed', 'completed_at': datetime.utcnow().isoformat()}
            update_data_clips['last_step_completed'] = self.name # Also update top-level last_step_completed
        elif status in ['failed', 'error']:
            step_status_entry = {'status': status, 'error': error_msg, 'failed_at': datetime.utcnow().isoformat()}
        else: # 'starting'
            step_status_entry = {'status': status, 'started_at': datetime.utcnow().isoformat()}
        
        existing_status_json[self.name] = step_status_entry
        update_data_clips['processing_status'] = existing_status_json
        
        try:
            await self.supabase.table('clips')\
                .update(update_data_clips)\
                .eq('id', video_id)\
                .eq('user_id', user_id)\
                .execute()
            self.logger.debug("Updated clips table progress", video_id=video_id, data=update_data_clips)
        except Exception as e:
            self.logger.error("Failed to update clips table progress", video_id=video_id, error=e, data_sent=update_data_clips)

        # Log processing event to 'processing_events' table
        try:
            await self.supabase.table('processing_events').insert({
                'video_id': video_id,
                'user_id': user_id,
                'step_name': self.name,
                'step_index': step_index, # Store 1-based index
                'status': status,
                'error': error_msg, # Renamed from 'error' in plan to 'error_msg' to avoid conflict
                'timestamp': datetime.utcnow().isoformat(),
                'metadata': {'total_steps': total_steps} # Example metadata
            }).execute()
            self.logger.debug("Logged processing event", video_id=video_id, step_name=self.name, status=status)
        except Exception as e:
            self.logger.error("Failed to log processing event", video_id=video_id, error=e)

    async def _save_step_results(self, video_id: str, user_id: Optional[str], result: StepResult):
        """Save step results immediately to database with user context."""
        await self._ensure_supabase_client()
        if not user_id:
            self.logger.warning("user_id not provided for saving step results, skipping database update.", video_id=video_id)
            return

        if not result.data: # Only proceed if there's data to save
            self.logger.debug("No data in StepResult to save.", video_id=video_id, step_name=self.name)
            return

        direct_update_payload = {}
        jsonb_updates = {}
        
        # Known JSONB columns in 'clips' table that steps might update
        # Note: 'processing_status' here refers to the JSONB column, not the main TEXT enum status.
        # The main TEXT enum status is handled by _update_progress or by the special key below.
        jsonb_columns = ["technical_metadata", "camera_details", "audio_tracks", "subtitle_tracks", "processing_status"]

        for key, value in result.data.items():
            if key in jsonb_columns and isinstance(value, dict):
                jsonb_updates[key] = value
            else:
                direct_update_payload[key] = value
        
        # Handle special key for overriding the main TEXT processing_status enum
        if "processing_status" in jsonb_updates and isinstance(jsonb_updates["processing_status"], dict):
            if "_overall_status_enum_override_" in jsonb_updates["processing_status"]:
                direct_update_payload["processing_status"] = jsonb_updates["processing_status"].pop("_overall_status_enum_override_")


        direct_update_payload['updated_at'] = datetime.utcnow().isoformat() # Changed from _last_updated
        # '_updated_by_step' could be part of a JSONB log or a separate events table,
        # for now, we'll log it. BaseStep._update_progress already logs step name.
        self.logger.debug(f"Step {self.name} is providing data for update.", video_id=video_id, direct_keys=list(direct_update_payload.keys()), jsonb_keys=list(jsonb_updates.keys()))

        try:
            # Fetch existing JSONB data to merge
            if jsonb_updates:
                select_columns = ", ".join(jsonb_updates.keys())
                existing_data_resp = await self.supabase.table('clips')\
                    .select(select_columns)\
                    .eq('id', video_id)\
                    .eq('user_id', user_id)\
                    .single().execute()

                if existing_data_resp.data:
                    for col_name, new_json_data in jsonb_updates.items():
                        existing_json = existing_data_resp.data.get(col_name, {})
                        if not isinstance(existing_json, dict): # Ensure it's a dict
                            existing_json = {}
                        existing_json.update(new_json_data) # Merge new data into existing
                        direct_update_payload[col_name] = existing_json
                else: # No existing data, just use the new JSONB data
                    for col_name, new_json_data in jsonb_updates.items():
                        direct_update_payload[col_name] = new_json_data
            
            if not direct_update_payload:
                self.logger.info(f"No direct data to update for step {self.name}", video_id=video_id)
            else:
                await self.supabase.table('clips')\
                    .update(direct_update_payload)\
                    .eq('id', video_id)\
                    .eq('user_id', user_id)\
                    .execute()
                self.logger.info(f"Saved results for step {self.name}", video_id=video_id, updated_keys=list(direct_update_payload.keys()))

        except Exception as e:
            self.logger.error(f"Failed to save step results to clips table for {self.name}", video_id=video_id, error=e, data_sent=direct_update_payload)

        # If there are artifacts (files created), store their paths
        if result.artifacts:
            artifact_records = []
            for key, path in result.artifacts.items():
                artifact_records.append({
                    'video_id': video_id,
                    'user_id': user_id,
                    'step_name': self.name,
                    'artifact_type': key,
                    'file_path': str(path), # Ensure path is string
                    'created_at': datetime.utcnow().isoformat()
                })
            try:
                await self.supabase.table('artifacts').insert(artifact_records).execute()
                self.logger.info(f"Saved {len(artifact_records)} artifacts for step {self.name}", video_id=video_id)
            except Exception as e:
                 self.logger.error(f"Failed to save artifacts for step {self.name}", video_id=video_id, error=e, artifacts_data=artifact_records)
    
    async def save_partial_result(self, video_id: str, user_id: Optional[str], partial_data: Dict[str, Any]):
        """Save partial results during processing (for long-running steps)."""
        await self._ensure_supabase_client()
        if not self.save_partial: # Check step's config
            return
        if not user_id:
            self.logger.warning("user_id not provided for saving partial results, skipping.", video_id=video_id)
            return

        update_payload = partial_data.copy()
        update_payload['_partial_update'] = True # Mark as partial
        update_payload['_partial_update_at'] = datetime.utcnow().isoformat()
        update_payload['_partial_updated_by_step'] = self.name


        try:
            await self.supabase.table('clips')\
                .update(update_payload)\
                .eq('id', video_id)\
                .eq('user_id', user_id)\
                .execute()
            self.logger.debug(f"Saved partial result for step {self.name}", video_id=video_id, data_keys=list(update_payload.keys()))
        except Exception as e:
            self.logger.error(f"Failed to save partial result for step {self.name}", video_id=video_id, error=e, data_sent=update_payload)
    
    async def setup(self):
        """Optional setup method called once when step is initialized by the pipeline."""
        # Subclasses can override this for one-time setup (e.g., loading models)
        self.logger.debug(f"Setup method called for step {self.name}")
        pass
    
    async def cleanup(self, video_id: str, result: StepResult):
        """Optional cleanup after processing a specific video."""
        # Subclasses can override this for cleanup related to a specific video processing
        self.logger.debug(f"Cleanup method called for step {self.name}", video_id=video_id, success=result.success)
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get step information, including its configuration."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category,
            "requires": self.requires,
            "optional_requires": self.optional_requires,
            "provides": self.provides,
            "config": self.config.model_dump() # Use model_dump for Pydantic v2
        }