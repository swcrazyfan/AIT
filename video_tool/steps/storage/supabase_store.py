import os
import structlog
from typing import Dict, Any

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata, StepConfig
from video_tool.core.enhanced_models import EnhancedVideoIngestOutput
# from video_tool.core.db import get_supabase # Supabase client is on BaseStep

logger = structlog.get_logger(__name__)

class SupabaseStoreStep(BaseStep):
    """
    Stores the consolidated video processing output into Supabase.
    """
    
    name = "supabase_store" # As per architecture diagram
    version = "1.0"
    description = "Store final processed video data into Supabase database."
    category = "storage"
    
    # This step typically runs last and requires the fully populated VideoIngestOutput model
    # or at least the video_id and the data to be stored.
    # This step runs last, relying on previous steps to have populated VideoMetadata.
    # It primarily finalizes the record.
    requires = ["video_id", "user_id"]
    provides = ["storage_finalized_status"] # Indicates finalization

    def __init__(self, config: StepConfig):
        super().__init__(config)
        # self.supabase_client is inherited from BaseStep and initialized in its execute method

    async def process(self, video: VideoMetadata, context: dict = None) -> StepResult:
        if not self.supabase: # self.supabase is initialized in BaseStep.execute()
            logger.error("Supabase client not available for SupabaseStoreStep.")
            return StepResult(
                success=False, step_name=self.name, video_id=video.video_id,
                error="Supabase client not initialized."
            )

        video_id = video.video_id
        user_id = video.user_id # Assuming user_id is part of VideoMetadata

        if not user_id:
            logger.error(f"User ID not found in video metadata for video_id: {video_id}")
            return StepResult(
                success=False, step_name=self.name, video_id=video_id,
                error="User ID missing in video metadata, cannot store to Supabase."
            )
            
        try:
            # This step's primary role is to finalize the 'clips' record status.
            # Individual data fields and related tables (like 'analysis', 'artifacts')
            # are expected to be populated by their respective steps and BaseStep._save_step_results.

            final_status_update = {
                "processing_status": ProcessingStatus.COMPLETED.value,
                "processing_progress": 100,
                "updated_at": datetime.utcnow().isoformat(),
                # 'processed_at' could also be set here if it signifies the end of all processing
                "processed_at": datetime.utcnow().isoformat()
            }
            
            # Ensure the processing_status JSONB also reflects completion for all enabled steps
            # This might be complex if not all steps ran or some were optional.
            # For now, we assume the pipeline ensures 'processed_steps' in VideoMetadata is accurate.
            
            # Fetch current 'processing_status' JSONB to update it carefully
            current_clip_data_resp = await self.supabase.table('clips')\
                .select('processing_status')\
                .eq('id', video_id)\
                .eq('user_id', user_id)\
                .single().execute()

            current_processing_status_json = {}
            if current_clip_data_resp.data and current_clip_data_resp.data.get('processing_status'):
                current_processing_status_json = current_clip_data_resp.data['processing_status']
                if not isinstance(current_processing_status_json, dict): # Ensure it's a dict
                    current_processing_status_json = {"error": "Invalid existing status format"}


            # Mark this storage step as completed within the JSONB status
            current_processing_status_json[self.name] = {
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat()
            }
            final_status_update["processing_status_details"] = current_processing_status_json # Use a different key for JSONB

            await self.supabase.table('clips').update(final_status_update)\
                .eq('id', video_id)\
                .eq('user_id', user_id)\
                .execute()

            logger.info(f"Finalized storage for video_id: {video_id}. Status set to COMPLETED.")

            return StepResult(
                success=True, step_name=self.name, video_id=video_id,
                data={"storage_finalized_status": ProcessingStatus.COMPLETED.value}
            )

        except Exception as e:
            logger.error(f"SupabaseStoreStep failed for video_id {video_id}: {str(e)}", exc_info=True)
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video_id,
                error=str(e)
            )

    async def setup(self):
        logger.info("SupabaseStoreStep initialized.")
        if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_ANON_KEY"):
            logger.warning("Supabase URL or Anon Key not configured in environment for SupabaseStoreStep.")