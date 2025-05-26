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
        for step_config_data in self.config.steps:
            # Ensure step_config_data is a dict if it comes from Pydantic model
            if not isinstance(step_config_data, dict):
                step_config_dict = step_config_data.dict()
            else:
                step_config_dict = step_config_data

            config = StepConfig(**step_config_dict.get('config', {}))

            step = self.registry.create_step(
                step_config_dict['category'],
                step_config_dict['name'],
                config
            )

            if not step:
                raise ValueError(
                    f"Step not found: {step_config_dict['category']}.{step_config_dict['name']}"
                )

            await step.setup()
            self.steps.append(step)

        # UPDATED: Register Procrastinate tasks using current patterns
        await self._register_tasks()

        # Register monitoring tasks
        await self._register_monitoring_tasks()

        self._initialized = True

    async def _register_tasks(self):
        """
        Tasks are now registered centrally in video_tool.procrastinate_app.
        This method ensures the pipeline's app instance is the shared one.
        Individual steps no longer need a direct .task attribute in the same way.
        The defer calls will use the named task from the app.
        """
        if not self.app: # Should have been set in initialize
            self.app = await get_procrastinate_app()
        
        # Verify the main task is registered on this app instance (optional sanity check)
        registered_task_name = "video_tool.tasks.run_pipeline_step_job"
        if registered_task_name not in self.app.tasks:
            logger.error(f"Critical: Task '{registered_task_name}' not found in Procrastinate app. Ensure it's registered in procrastinate_app.py.")
            # This might indicate a problem with app instance sharing or registration.
            # For now, we assume it's registered.
        else:
            logger.info(f"Confirmed task '{registered_task_name}' is registered on the app instance.")


    async def _register_monitoring_tasks(self):
        """Register periodic monitoring tasks"""

        @self.app.periodic(cron="*/5 * * * *")  # Every 5 minutes
        @self.app.task(name="video_tool.monitoring.check_stalled_videos", queue="monitoring") # Added @self.app.task
        async def check_stalled_videos():
            """Find and restart stalled video processing"""
            stalled = self.supabase.table('clips')\
                .select('id, current_step, updated_at')\
                .lt('processing_progress', 100)\
                .lt('updated_at', (datetime.utcnow() - timedelta(minutes=30)).isoformat())\
                .execute() # No await

            for video_data in stalled.data:
                logger.warning(f"Found stalled video: {video_data['id']}")
                await self.resume_video(video_data['id'])

        @self.app.periodic(cron="0 * * * *")  # Every hour
        @self.app.task(name="video_tool.monitoring.cleanup_old_artifacts", queue="monitoring") # Added @self.app.task
        async def cleanup_old_artifacts():
            """Clean up artifacts from completed videos older than 7 days"""
            old_artifacts = self.supabase.table('artifacts')\
                .select('*')\
                .lt('created_at', (datetime.utcnow() - timedelta(days=7)).isoformat())\
                .execute() # No await

            for artifact in old_artifacts.data:
                # Delete file if exists
                file_path = Path(artifact['file_path'])
                if file_path.exists():
                    file_path.unlink()

                # Remove record
                self.supabase.table('artifacts')\
                    .delete()\
                    .eq('id', artifact['id'])\
                    .execute() # No await

    async def process_video(self, file_path: str, auth_manager: 'AuthManager') -> str:
        """Start processing a video through the pipeline with authentication"""
        if not self._initialized:
            await self.initialize()

        # Get authenticated user
        session = await auth_manager.get_current_session() # Added await
        if not session:
            raise ValueError("Authentication required")
        
        # Ensure the pipeline's Supabase client is using the authenticated session
        try:
            self.supabase.auth.set_session(
                access_token=session['access_token'],
                refresh_token=session['refresh_token']
            )
            logger.info(f"User session set for pipeline's Supabase client. User: {session.get('email', session.get('user_id'))}")
        except Exception as e:
            logger.error(f"Failed to set session on pipeline's Supabase client: {e}", exc_info=True)
            # Depending on the desired behavior, you might want to raise an error here
            # or attempt to proceed if some operations can be done anonymously (though unlikely for inserts).
            raise ValueError(f"Could not set authenticated session for Supabase client: {e}")


        user_id = session['user_id']

        # Check for existing video with same path
        existing_result = self.supabase.table('clips')\
            .select('id, file_checksum, processing_status')\
            .eq('file_path', file_path)\
            .eq('user_id', user_id)\
            .execute() # No await, confirmed by previous run

        if existing_result.data:
            video_id = existing_result.data[0]['id']
            status_str = existing_result.data[0]['processing_status'] # Changed to processing_status
            # Ensure status is a valid ProcessingStatus enum member or string
            try:
                # If processing_status is a JSONB object like {'status': 'completed'}, extract it.
                # This depends on how VideoMetadata Pydantic model expects 'status'
                # Assuming status_str might be the direct enum value or a dict containing it.
                if isinstance(status_str, dict):
                    status_val = status_str.get('status', status_str) # Get 'status' field or use as is
                else:
                    status_val = status_str
                status = ProcessingStatus(status_val)
            except ValueError:
                logger.warning(f"Invalid status '{status_str}' for video {video_id}, treating as new.")
                status = ProcessingStatus.QUEUED # Default to queued or handle as error

            if status == ProcessingStatus.COMPLETED:
                logger.info(f"Video already processed: {video_id}")
                return video_id
            else:
                logger.info(f"Resuming processing for video: {video_id}")
                await self.resume_video(video_id)
                return video_id

        # Create initial video record
        from uuid import uuid4
        video = VideoMetadata(
            video_id=str(uuid4()),
            file_path=file_path,
            file_name=Path(file_path).name,
            user_id=user_id,
            processing_status=ProcessingStatus.QUEUED.value, # Ensure using the value for DB
            # Ensure steps is not empty before accessing config
            total_steps=len([s for s in self.steps if s.config and s.config.enabled]) if self.steps else 0
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
            next_step_obj = self.steps[next_idx]
            if next_step_obj.config and next_step_obj.config.enabled:
                logger.info(f"Resuming: Queuing step {next_step_obj.name} for video {video_id}")
                job_id = await self.app.tasks["video_tool.tasks.run_pipeline_step_job"].defer_async(
                    video_id=video_id,
                    pipeline_config_path=self.config_path,
                    step_category=next_step_obj.category,
                    step_name=next_step_obj.name,
                    queue=next_step_obj.config.queue # Pass the step's configured queue
                )
                logger.info(f"Resuming: Queued job {job_id} for step {next_step_obj.name} on queue '{next_step_obj.config.queue}' for video {video_id}")
            else:
                logger.info(f"Resuming: Next step {next_step_obj.name} is not enabled or no config. Video {video_id}")
        else:
            logger.info(f"Resuming: No more steps to queue for video {video_id}. Last completed index was {last_completed_idx}.")


    async def _queue_first_step(self, video_id: str):
        """Queue the first enabled step using the generic task."""
        for step in self.steps: # Iterate through step objects
            if step.config and step.config.enabled:
                logger.info(f"Queueing first step {step.name} for video {video_id}")
                job_id = await self.app.tasks["video_tool.tasks.run_pipeline_step_job"].defer_async(
                    video_id=video_id,
                    pipeline_config_path=self.config_path, # Pass config path
                    step_category=step.category,
                    step_name=step.name,
                    queue=step.config.queue # Pass the step's configured queue
                )
                logger.info(f"Queued first job {job_id} for step {step.name} on queue '{step.config.queue}' for video {video_id}")
                return # Queue only the first enabled step
        logger.warning(f"No enabled steps found to queue for video {video_id}")


    async def _queue_next_step(self, video_id: str, current_step_idx: int):
        """Queue the next enabled step using the generic task."""
        next_step_idx = current_step_idx + 1
        if next_step_idx < len(self.steps):
            for i in range(next_step_idx, len(self.steps)):
                step = self.steps[i]
                if step.config and step.config.enabled:
                    logger.info(f"Queueing next step {step.name} for video {video_id}")
                    job_id = await self.app.tasks["video_tool.tasks.run_pipeline_step_job"].defer_async(
                        video_id=video_id,
                        pipeline_config_path=self.config_path, # Pass config path
                        step_category=step.category,
                        step_name=step.name,
                        queue=step.config.queue # Pass the step's configured queue
                    )
                    logger.info(f"Queued next job {job_id} for step {step.name} on queue '{step.config.queue}' for video {video_id}")
                    return # Queue only the next enabled step
            logger.info(f"No more enabled steps to queue after step index {current_step_idx} for video {video_id}")
        else:
            logger.info(f"All steps processed for video {video_id} after step index {current_step_idx}.")


    async def _get_video_metadata(self, video_id: str) -> Optional[VideoMetadata]:
        """Get current video metadata from database"""
        # execute() is synchronous
        result = self.supabase.table('clips')\
            .select('*')\
            .eq('id', video_id)\
            .single()\
            .execute() # No await

        if result.data:
            return VideoMetadata(**result.data)
        return None

    async def _create_video_record(self, video: VideoMetadata):
        """Create initial video record"""
        # Explicitly map VideoMetadata fields to clips table columns
        data_to_insert = {
            'id': video.video_id, # Map video_id (from Pydantic) to id (DB column)
            'user_id': video.user_id,
            'file_path': video.file_path,
            'local_path': video.file_path, # Default local_path to file_path for initial creation
            'file_name': video.file_name,
            # Optional fields from VideoMetadata, map if present
            'file_checksum': video.checksum if video.checksum is not None else None,
            'file_size_bytes': video.file_size_bytes if video.file_size_bytes is not None else None,
            'duration_seconds': video.duration_seconds if video.duration_seconds is not None else None,
            'width': video.width if video.width is not None else None,
            'height': video.height if video.height is not None else None,
            'frame_rate': video.fps if video.fps is not None else None, # Map Pydantic 'fps' to DB 'frame_rate'
            'codec': video.codec if video.codec is not None else None,
            'processing_status': video.status.value if video.status is not None else ProcessingStatus.QUEUED.value, # Map Pydantic 'status' to DB 'processing_status'
            'content_tags': video.tags if video.tags else [], # Map Pydantic 'tags' to DB 'content_tags'
            'thumbnails': video.thumbnails if video.thumbnails else [],
            'processed_steps': video.processed_steps if video.processed_steps else [],
            'created_at': video.created_at.isoformat() if video.created_at else datetime.utcnow().isoformat(),
            'updated_at': video.updated_at.isoformat() if video.updated_at else datetime.utcnow().isoformat(),
            'total_steps': video.total_steps if hasattr(video, 'total_steps') else 0, # From VideoMetadata
            # Provide temporary placeholders for NOT NULL fields if they are None
            'file_checksum': video.checksum if video.checksum is not None else f"pending_checksum_{video.video_id}",
            'file_size_bytes': video.file_size_bytes if video.file_size_bytes is not None else 0
        }
        
        # Handle 'container' if VideoMetadata has 'format_name'
        if hasattr(video, 'format_name') and video.format_name is not None:
            data_to_insert['container'] = video.format_name

        # Ensure critical NOT NULL fields have values (though some are defaulted above)
        if not data_to_insert.get('id'):
            raise ValueError("Cannot insert clip: 'id' (from video.video_id) is missing.")
        if not data_to_insert.get('user_id'):
            raise ValueError(f"Cannot insert clip {data_to_insert.get('id')}: 'user_id' is missing.")
        if not data_to_insert.get('file_path'):
            logger.error(f"file_path missing for clip {data_to_insert.get('id')}, using placeholder.") # Use global logger
            data_to_insert['file_path'] = "unknown_path_placeholder"
        if not data_to_insert.get('local_path'): # Should be set from file_path
             data_to_insert['local_path'] = data_to_insert['file_path']
        if not data_to_insert.get('file_name'):
            logger.error(f"file_name missing for clip {data_to_insert.get('id')}, using placeholder.") # Use global logger
            data_to_insert['file_name'] = "unknown_name_placeholder"

        # Warnings are still useful if actual None values were intended to be caught by earlier logic
        if video.checksum is None: # Check original video object's property
            logger.warning(f"Original video.checksum is None for video {data_to_insert.get('id')}. Using placeholder for initial insert.")
        if video.file_size_bytes is None: # Check original video object's property
            logger.warning(f"Original video.file_size_bytes is None for video {data_to_insert.get('id')}. Using placeholder for initial insert.")

        # execute() is synchronous
        self.supabase.table('clips').insert(data_to_insert).execute() # No await


    async def _mark_step_completed(self, video_id: str, step_name: str):
        """Mark a step as completed"""
        # Get current processed steps
        # execute() is synchronous
        result = self.supabase.table('clips')\
            .select('processed_steps')\
            .eq('id', video_id)\
            .single()\
            .execute() # No await

        processed_steps = result.data.get('processed_steps', []) if result.data else []
        if step_name not in processed_steps:
            processed_steps.append(step_name)

        # execute() is synchronous
        self.supabase.table('clips').update({
            'processed_steps': processed_steps,
            'last_step_completed': step_name,
            'updated_at': datetime.utcnow().isoformat()
        }).eq('id', video_id).execute() # No await

    async def _mark_step_failed(self, video_id: str, step_name: str, error: Optional[str]):
        """Mark a step as failed"""
        # execute() is synchronous
        self.supabase.table('clips').update({
            'processing_status': ProcessingStatus.FAILED.value, # Changed to processing_status
            'current_step': step_name,
            'error_details': f"{step_name}: {error if error else 'Unknown error'}",
            'updated_at': datetime.utcnow().isoformat()
        }).eq('id', video_id).execute() # No await

    async def _mark_pipeline_completed(self, video_id: str):
        """Mark entire pipeline as completed"""
        # execute() is synchronous
        self.supabase.table('clips').update({
            'processing_status': ProcessingStatus.COMPLETED.value, # Changed to processing_status
            'processing_progress': 100,
            'updated_at': datetime.utcnow().isoformat(),
            # 'completed_at': datetime.utcnow().isoformat()
        }).eq('id', video_id).execute() # No await

class PipelineMonitor:
    """Real-time pipeline monitoring"""

    def __init__(self, supabase):
        self.supabase = supabase

    async def get_active_videos(self) -> List[Dict]:
        """Get all actively processing videos"""
        result = self.supabase.table('active_processing')\
            .select('*')\
            .execute() # No await
        return result.data if result.data else []

    async def get_video_progress(self, video_id: str) -> Dict:
        """Get detailed progress for a specific video"""
        video_result = self.supabase.table('clips')\
            .select('*, processing_events(*)')\
            .eq('id', video_id)\
            .single()\
            .execute() # No await

        if not video_result.data:
            return {}

        # Calculate step timings
        events = video_result.data.get('processing_events', [])
        step_timings = {}

        for event in sorted(events, key=lambda e: e['timestamp']):
            step = event['step_name']
            if step not in step_timings:
                step_timings[step] = {}

            if event['status'] == 'starting': # Assuming 'starting' status from BaseStep._update_progress
                step_timings[step]['start'] = event['timestamp']
            elif event['status'] == 'completed':
                step_timings[step]['end'] = event['timestamp']
                if 'start' in step_timings[step] and step_timings[step]['start']:
                    try:
                        start = datetime.fromisoformat(step_timings[step]['start'].replace('Z', '+00:00'))
                        end = datetime.fromisoformat(step_timings[step]['end'].replace('Z', '+00:00'))
                        step_timings[step]['duration'] = (end - start).total_seconds()
                    except ValueError as e:
                        logger.warning(f"Could not parse timestamp for step timing: {e}")
                        step_timings[step]['duration'] = None


        return {
            'video': video_result.data,
            'step_timings': step_timings,
            'total_duration': sum(s.get('duration', 0) for s in step_timings.values() if s.get('duration') is not None)
        }

    async def watch_video(self, video_id: str, callback=None):
        """Watch a video's progress in real-time"""
        last_progress = -1

        while True:
            progress_data = await self.get_video_progress(video_id)
            if not progress_data.get('video'):
                logger.warning(f"Video {video_id} not found for monitoring.")
                break

            current_progress = progress_data['video'].get('processing_progress', 0)

            if current_progress != last_progress:
                if callback:
                    await callback(progress_data)
                else:
                    print(f"Progress: {current_progress}% - {progress_data['video'].get('current_step')}")

                last_progress = current_progress

            status_val_from_monitor = progress_data['video'].get('processing_status') # Changed to processing_status
            try:
                # Handle if status_val_from_monitor is a dict or direct value
                if isinstance(status_val_from_monitor, dict):
                    actual_status_str = status_val_from_monitor.get('status', status_val_from_monitor)
                else:
                    actual_status_str = status_val_from_monitor
                
                status = ProcessingStatus(actual_status_str) if actual_status_str else None
            except ValueError:
                status = None # Or handle unknown status

            if current_progress >= 100 or status == ProcessingStatus.FAILED:
                break

            await asyncio.sleep(2)