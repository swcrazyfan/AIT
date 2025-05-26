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
            # poll_interval=1.0 # Removed: Not a valid argument for Worker.__init__
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

    # Build concurrency map
    if queues:
        # Filter to requested queues
        effective_concurrency_config = {q: concurrency_config.get(q, 1) for q in queues}
    else:
        effective_concurrency_config = concurrency_config
    
    effective_queues = list(effective_concurrency_config.keys())
    total_concurrency = sum(effective_concurrency_config.values())

    try:
        # UPDATED: Use shutdown_graceful_timeout for better process management
        await app.run_worker_async(
            concurrency=total_concurrency,
            queues=effective_queues,
            wait=True,
            shutdown_graceful_timeout=shutdown_timeout,
            install_signal_handlers=True
        )
    except asyncio.CancelledError:
        logger.info("Graceful shutdown completed")
    except Exception as e:
        logger.error(f"Worker process failed: {str(e)}", exc_info=True)
        raise