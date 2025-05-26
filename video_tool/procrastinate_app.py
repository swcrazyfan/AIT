from dotenv import load_dotenv
load_dotenv() # Ensure .env is loaded before db imports

from procrastinate import App, PsycopgConnector
import os
import structlog

from video_tool.tasks import run_pipeline_step_job # Import the task function

logger = structlog.get_logger(__name__)

# This is the Procrastinate App instance that the CLI tool will discover.
# It needs to be configured similarly to how it's done in video_tool.core.db.Database.initialize_procrastinate

db_url = os.getenv("SUPABASE_DB_URL")
if not db_url:
    logger.error("SUPABASE_DB_URL must be set in .env for Procrastinate app instantiation.")
    raise ValueError("SUPABASE_DB_URL must be set in .env for Procrastinate app instantiation.")

connector = PsycopgConnector(conninfo=db_url)
app = App(
    connector=connector,
    import_paths=["video_tool.tasks", "video_tool.steps", "video_tool.core"]
    # app_name argument removed as it's not supported directly in constructor for this version
)

# Register the generic pipeline step task
# The name here is critical and will be used when deferring tasks.
# It must be the full Python path to the function.
app.task(
    _func=run_pipeline_step_job,  # Pass the function directly
    name="video_tool.tasks.run_pipeline_step_job", # Explicitly define the importable name
    queue="default" # Or make it configurable, or steps define their own queues
)

logger.info("Procrastinate app defined in procrastinate_app.py and task registered.")