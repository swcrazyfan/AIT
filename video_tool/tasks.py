import structlog

# Import the procrastinate app instance.
# This assumes get_procrastinate_app() returns the globally shared app instance.
from video_tool.core.db import get_procrastinate_app

# We need to be careful with when get_procrastinate_app() is called.
# If it's called at import time here, it might trigger initialization too early
# or lead to circular dependencies if db.py also imports something from tasks.py (not the case here).
# A common pattern is to have a central app definition, e.g., in procrastinate_app.py
# and import that. For now, let's proceed with get_procrastinate_app.

# app = get_procrastinate_app() # Calling awaitable at import time is not ideal.
# Instead, the app object should be fetched when needed, or tasks registered via CLI.
# For simplicity in this step, we'll assume the app object is available
# or tasks are registered on an app instance that the worker also uses.

# The @app.task decorator needs an app instance.
# Let's define the function and it will be decorated where the app is available,
# or we retrieve the app instance here.

logger = structlog.get_logger(__name__)

async def actual_step_execution_logic(video_id: str, pipeline_config_path: str, step_category: str, step_name: str, job_context: dict):
    """
    The core logic for executing a step.
    This is extracted to be callable from the Procrastinate task.
    """
    from video_tool.core.pipeline import Pipeline # Local import to avoid circular dependencies at import time

    logger.info(
        "Executing pipeline step",
        video_id=video_id,
        pipeline_config_path=pipeline_config_path,
        step_category=step_category,
        step_name=step_name,
        procrastinate_job_context=job_context
    )

    pipeline = Pipeline(pipeline_config_path)
    # Initialize only if not already, but pipeline.initialize() is idempotent
    await pipeline.initialize()

    # Find the step
    current_step = None
    step_idx = -1
    for idx, s in enumerate(pipeline.steps):
        if s.category == step_category and s.name == step_name:
            current_step = s
            step_idx = idx
            break
    
    if not current_step:
        logger.error(
            "Step not found in pipeline",
            video_id=video_id,
            step_category=step_category,
            step_name=step_name,
            pipeline_config_path=pipeline_config_path
        )
        raise ValueError(f"Step {step_category}.{step_name} not found in pipeline {pipeline_config_path}")

    logger.info(f"Found step: {current_step.name} at index {step_idx} for video_id: {video_id}")

    video = await pipeline._get_video_metadata(video_id)
    if not video:
        logger.error(f"Video not found for processing: {video_id}", step_name=current_step.name)
        raise ValueError(f"Video {video_id} not found, cannot process step {current_step.name}")

    try:
        if current_step.name in video.processed_steps:
            logger.info(f"Step {current_step.name} already completed for {video_id}, skipping.")
            await pipeline._queue_next_step(video_id, step_idx)
            return {"status": "skipped", "reason": "already_completed"}

        # Pass the Procrastinate job context to the step's execute method
        # The step.execute method expects: video, step_num, total_steps, context
        result = await current_step.execute(
            video,
            step_idx + 1, # Human-readable step number
            len(pipeline.steps),
            job_context # Pass the Procrastinate job context dictionary
        )

        if result.success:
            await pipeline._mark_step_completed(video_id, current_step.name)
            if step_idx + 1 < len(pipeline.steps):
                await pipeline._queue_next_step(video_id, step_idx)
            else:
                await pipeline._mark_pipeline_completed(video_id)
        else:
            await pipeline._mark_step_failed(video_id, current_step.name, result.error)
            if current_step.config.params.get('continue_on_failure', False):
                await pipeline._queue_next_step(video_id, step_idx)
        
        return result.dict()

    except Exception as e:
        logger.error(f"Task execution failed for step {current_step.name}, video {video_id}", exc_info=True)
        # Ensure _mark_step_failed uses the same pipeline instance / app
        await pipeline._mark_step_failed(video_id, current_step.name, str(e))
        raise


# This is the function that will be decorated with @app.task
# The Procrastinate app instance needs to be available when this module is imported by the worker,
# or when the task is explicitly registered.
# Let's assume the app is retrieved and the task is registered in procrastinate_app.py or similar central place.

# For now, this function is defined. The @app.task decoration will be handled
# in a place where `app` is definitively initialized, like `procrastinate_app.py`.
async def run_pipeline_step_job(context, video_id: str, pipeline_config_path: str, step_category: str, step_name: str):
    """
    Procrastinate job entry point. `context` is Procrastinate's JobContext.
    """
    # The `context` object from Procrastinate contains job details.
    # We can pass context.job.as_dict() or specific attributes if needed by actual_step_execution_logic
    job_context_dict = context.job.as_dict() if context and hasattr(context, 'job') else {}
    return await actual_step_execution_logic(video_id, pipeline_config_path, step_category, step_name, job_context_dict)