import asyncio
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
import yaml
import structlog # Import structlog

from video_tool.core.pipeline import Pipeline, PipelineMonitor
from video_tool.core.registry import StepRegistry
from video_tool.core.db import get_procrastinate_app, get_supabase
from video_tool.core.auth import AuthManager
from video_tool.core.search import VideoSearcher, format_search_results, format_duration # Import format_duration
from video_tool.worker import run_workers

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


app = typer.Typer(help="Modular Video Processing Tool")
console = Console()
logger = structlog.get_logger(__name__) # Define logger for this module

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
    pattern: str = typer.Option("*.mp4,*.MP4,*.mov,*.MOV,*.avi,*.AVI", "--pattern", "-p"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be processed")
):
    """Ingest videos using configured pipeline (requires authentication)"""
    
    async def run():
        # Check authentication
        auth_manager = AuthManager()
        session = await auth_manager.get_current_session() # Await async call
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
            
            procrastinate_app = await get_procrastinate_app()
            
            # Run workers
            await run_workers(procrastinate_app, workers, pipeline.config.worker_config)
        else:
            console.print("\n[dim]Run workers separately with: video-tool worker[/dim]")
        
        # If watching, start watcher
        if watch:
            if not WATCHDOG_AVAILABLE:
                console.print("[red]Watchdog library not installed. Cannot watch directory.[/red]")
                console.print("Install with: [cyan]pip install watchdog[/cyan]")
                return
            console.print(f"\n[cyan]Watching {path} for new files...[/cyan]")
            await watch_directory(path, pipeline, pattern, recursive, auth_manager)
    
    # Run async function
    asyncio.run(run())

# Authentication commands
@auth_app.command("login")
def auth_login(
    email_opt: Optional[str] = typer.Option(None, "--email", "-e", help="Email address"),
    password_opt: Optional[str] = typer.Option(None, "--password", "-p", help="Password")
):
    """Login to your account"""
    
    async def run(email_from_option, password_from_option):
        auth_manager = AuthManager()
        
        # Initialize current_email and current_password with values from options
        current_email = email_from_option
        current_password = password_from_option

        # Check if already logged in
        session = await auth_manager.get_current_session() # Await async call
        if session:
            console.print(f"[yellow]Already logged in as: {session['email']}[/yellow]")
            if not typer.confirm("Do you want to login with a different account?"):
                return
            else:
                # User wants to login with a different account, so force prompt by clearing current credentials
                current_email = None
                current_password = None
        
        # Get credentials if not provided (either initially or because user wants to re-login)
        if not current_email:
            current_email = Prompt.ask("Email")
        if not current_password:
            current_password = Prompt.ask("Password", password=True)
        
        # Attempt login
        with console.status("[bold green]Logging in..."):
            success = await auth_manager.login(current_email, current_password)
        
        if success:
            console.print("[green]✓ Successfully logged in![/green]")
            
            # Get and show user profile
            profile = await auth_manager.get_user_profile()
            if profile:
                console.print(f"Welcome, {profile.get('display_name', current_email)}!") # Use current_email
                if profile.get('profile_type') == 'admin':
                    console.print("[yellow]Admin access granted[/yellow]")
        else:
            console.print("[red]✗ Login failed. Please check your credentials.[/red]")
    
    asyncio.run(run(email_opt, password_opt))

@auth_app.command("signup")
def auth_signup(
    email_opt: Optional[str] = typer.Option(None, "--email", "-e", help="Email address", prompt=True, confirmation_prompt=False, hide_input=False),
    password_opt: Optional[str] = typer.Option(None, "--password", "-p", help="Password", prompt=True, confirmation_prompt=True, hide_input=True)
):
    """Create a new account"""
    
    async def run(email_arg, password_arg): # Pass arguments
        auth_manager = AuthManager()
        
        # Use passed arguments
        current_email = email_arg
        current_password = password_arg
        
        # If password was prompted by Typer due to confirmation, it's already confirmed.
        # If not prompted by Typer (e.g. passed via CLI option without prompt),
        # and we still need to confirm, that logic would be more complex here.
        # For simplicity, assuming Typer's prompt handles confirmation if password_opt was None.

        # Attempt signup
        with console.status("[bold green]Creating account..."):
            success = await auth_manager.signup(current_email, current_password)
        
        if success:
            console.print("[green]✓ Account created successfully![/green]")
            console.print("[yellow]Please check your email to confirm your account.[/yellow]")
        else:
            console.print("[red]✗ Signup failed. Email may already be registered.[/red]")
    
    # Pass the CLI options to the run function
    asyncio.run(run(email_opt, password_opt))

@auth_app.command("logout")
def auth_logout():
    """Logout from current session"""
    
    async def run():
        auth_manager = AuthManager()
        
        session = await auth_manager.get_current_session() # Await async call
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
        
        session = await auth_manager.get_current_session() # Await async call
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
            status_table.add_row("Created", profile.get('created_at', 'N/A')[:10] if profile.get('created_at') else 'N/A')
        
        console.print(status_table)
        
        # Get user stats
        client = await auth_manager.get_authenticated_client()
        if client:
            stats_result = await client.rpc('get_user_stats').execute()
            if stats_result.data and stats_result.data[0]: # Check if data is not empty and first element exists
                stats = stats_result.data[0]
                
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
            else:
                console.print("\n[yellow]No video processing statistics available.[/yellow]")
    
    asyncio.run(run())

# Search commands
@search_app.command("query")
def search_query_cmd( # Renamed to avoid conflict with the variable 'query'
    query_str: str = typer.Argument(..., help="Search query"),
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
                query=query_str, # Use the renamed argument
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
                        row.append(str(result.get('similarity_score', '0.000')))
                    elif search_type == "hybrid":
                        row.append(str(result.get('search_rank', '0.000')))
                        row.append(result.get('match_type', 'unknown'))
                    elif search_type == "fulltext":
                        row.append(str(result.get('fts_rank', '0.000')))
                
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
                    f"{float(result.get('similarity_score', 0)):.3f}"
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
            info_table.add_row("Duration", format_duration(clip.get('duration_seconds', 0))) # Use imported format_duration
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
            stats_result = await searcher.get_user_stats()
            
            if not stats_result or not stats_result[0]: # Check if data is not empty and first element exists
                console.print("[yellow]No statistics available[/yellow]")
                return

            stats = stats_result[0]
            
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
    workers_count: int = typer.Option(4, "--workers", "-w", help="Number of workers"), # Renamed to avoid conflict
    queues: Optional[List[str]] = typer.Option(None, "--queue", "-q", help="Specific queues to process"),
    config: str = typer.Option("default", "--config", "-c", help="Pipeline configuration")
):
    """Run Procrastinate workers"""
    
    async def run():
        # Load pipeline config to get worker settings
        config_path = Path("configs") / f"{config}.yaml"
        if not config_path.exists():
            console.print(f"[red]Configuration not found: {config_path}[/red]")
            return
            
        pipeline = Pipeline(str(config_path))
        await pipeline.initialize()
        
        procrastinate_app = await get_procrastinate_app()
        
        # Determine queues
        effective_queues = queues
        if not effective_queues:
            effective_queues = list(pipeline.config.worker_config.keys())
        
        # --- TEST: Force listening only on metadata queue ---
        effective_queues_test = ["metadata"]
        logger.warning(f"TESTING: Worker forced to listen only on queues: {effective_queues_test}. Original effective_queues: {effective_queues}")
        # --- END TEST ---

        console.print(f"[cyan]Starting {workers_count} workers for queues: {', '.join(effective_queues_test)}[/cyan]") # Use test queues
        
        # Pass the original worker_config for concurrency settings, but the filtered effective_queues_test for listening
        await run_workers(procrastinate_app, workers_count, pipeline.config.worker_config, effective_queues_test)
    
    asyncio.run(run())

@app.command()
def status(
    watch_status: bool = typer.Option(False, "--watch", "-w", help="Watch status continuously") # Renamed
):
    """Show processing status"""
    
    async def show_status_async(): # Renamed to avoid conflict
        # Check authentication
        auth_manager = AuthManager()
        client = await auth_manager.get_authenticated_client()
        if not client:
            console.print("[red]Authentication required. Please login first.[/red]")
            return
        
        procrastinate_app = await get_procrastinate_app()
        
        while True:
            # Get queue stats from Procrastinate
            jobs = await procrastinate_app.jobs.count_by_status() # Corrected: app -> procrastinate_app
            
            # Get user's video stats from Supabase
            current_session = await auth_manager.get_current_session() # Await async call
            if not current_session: # Should not happen if client is authenticated, but good practice
                console.print("[red]Session expired or invalid. Please login again.[/red]")
                return

            videos_result = await client.table('clips')\
                .select('status')\
                .eq('user_id', current_session['user_id'])\
                .execute()
                
            video_stats = {}
            if videos_result.data:
                for video_item in videos_result.data: # Corrected: video -> video_item
                    item_status = video_item['status'] # Corrected: status -> item_status
                    video_stats[item_status] = video_stats.get(item_status, 0) + 1
            
            # Clear screen if watching
            if watch_status: # Use renamed parameter
                console.clear()
            
            # Show job queue status
            queue_table = Table(title="Job Queue Status")
            queue_table.add_column("Status", style="cyan")
            queue_table.add_column("Count", style="yellow")
            
            for job_status, count in jobs.items(): # Corrected: status -> job_status
                queue_table.add_row(str(job_status), str(count)) # Ensure status is string
            
            console.print(queue_table)
            
            # Show video status
            video_table = Table(title="Your Video Processing Status")
            video_table.add_column("Status", style="cyan")
            video_table.add_column("Count", style="yellow")
            
            for item_status, count in sorted(video_stats.items()): # Corrected: status -> item_status
                video_table.add_row(str(item_status), str(count)) # Ensure status is string
            
            console.print(video_table)
            
            if not watch_status: # Use renamed parameter
                break
            
            await asyncio.sleep(2)
    
    asyncio.run(show_status_async()) # Use renamed async function

@app.command()
def monitor(
    video_id: Optional[str] = typer.Option(None, "--video", "-v", help="Monitor specific video"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow progress in real-time")
):
    """Monitor processing progress"""
    
    async def run_monitor():
        
        monitor_instance = PipelineMonitor(await get_supabase()) # Renamed to avoid conflict
        
        if video_id:
            # Monitor specific video
            if follow:
                # Real-time following
                async def progress_callback(progress):
                    console.clear()
                    
                    # Show video info
                    video_data = progress['video'] # Renamed to avoid conflict
                    info_table = Table(title=f"Video: {video_data['file_name']}")
                    info_table.add_column("Property", style="cyan")
                    info_table.add_column("Value", style="green")
                    
                    info_table.add_row("Status", str(video_data['status']))
                    info_table.add_row("Progress", f"{video_data['processing_progress']}%")
                    info_table.add_row("Current Step", video_data.get('current_step', 'N/A'))
                    info_table.add_row("Steps Completed", f"{len(video_data.get('processed_steps', []))}/{video_data.get('total_steps', 0)}")
                    
                    console.print(info_table)
                    
                    # Show step timings
                    if progress['step_timings']:
                        timing_table = Table(title="Step Timings")
                        timing_table.add_column("Step", style="cyan")
                        timing_table.add_column("Duration", style="yellow")
                        timing_table.add_column("Status", style="green")
                        
                        for step_name, timing in progress['step_timings'].items(): # Renamed step -> step_name
                            duration = f"{timing.get('duration', 0):.1f}s" if 'duration' in timing else "Running..."
                            timing_status = "✓" if 'end' in timing else "⏳" # Renamed status -> timing_status
                            timing_table.add_row(step_name, duration, timing_status)
                        
                        console.print(timing_table)
                    
                    # Show available features
                    features_table = Table(title="Available Features")
                    features_table.add_column("Feature", style="cyan")
                    features_table.add_column("Status", style="green")
                    
                    features_table.add_row("Metadata", "✓" if video_data.get('duration_seconds') else "✗")
                    features_table.add_row("Thumbnails", f"✓ ({video_data.get('thumbnail_count', 0)})" if video_data.get('thumbnails') else "✗")
                    features_table.add_row("Searchable", "✓" if video_data.get('embeddings') else "✗") # Assuming embeddings means searchable
                    features_table.add_row("Transcript", "✓" if video_data.get('transcript') else "✗")
                    
                    console.print(features_table)
                
                await monitor_instance.watch_video(video_id, progress_callback)
            else:
                # One-time progress check
                progress = await monitor_instance.get_video_progress(video_id)
                console.print(progress)
        else:
            # Show all active videos
            while True:
                console.clear()
                active = await monitor_instance.get_active_videos()
                
                if not active:
                    console.print("[yellow]No videos currently processing[/yellow]")
                else:
                    table = Table(title=f"Active Processing ({len(active)} videos)")
                    table.add_column("File", style="cyan", max_width=30)
                    table.add_column("Progress", style="green")
                    table.add_column("Step", style="yellow")
                    table.add_column("Status", style="magenta")
                    table.add_column("Time", style="blue")
                    
                    for video_data in active: # Renamed video -> video_data
                        progress_bar_val = video_data['processing_progress'] if video_data['processing_progress'] is not None else 0
                        progress_bar = f"[{'█' * (progress_bar_val // 10)}{'░' * (10 - progress_bar_val // 10)}]"
                        time_ago = f"{float(video_data.get('seconds_since_update', 0)):.0f}s ago"
                        
                        table.add_row(
                            video_data['file_name'],
                            f"{progress_bar} {progress_bar_val}%",
                            video_data.get('current_step', 'N/A'),
                            str(video_data['status']),
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
        
        for step_info in category_steps: # Renamed step -> step_info
            console.print(f"  [green]{step_info['name']}[/green] (v{step_info['version']})")
            console.print(f"    [dim]{step_info['description']}[/dim]")

# Helper functions
def find_video_files(path_obj: Path, recursive: bool, pattern_str: str) -> List[Path]: # Renamed parameters
    """Find video files matching pattern"""
    patterns = pattern_str.split(',')
    files = []
    
    if path_obj.is_file():
        # Check if the single file matches any of the patterns
        for p_item in patterns: # Renamed pattern -> p_item
            # A simple check, might need more robust glob-like matching for single files if patterns are complex
            if path_obj.name.endswith(p_item.strip().replace('*', '')): # Basic suffix check
                 files.append(path_obj)
                 break # Found a match, no need to check other patterns
        return files

    for p_item in patterns: # Renamed pattern -> p_item
        if recursive:
            files.extend(path_obj.rglob(p_item.strip()))
        else:
            files.extend(path_obj.glob(p_item.strip()))
    
    return sorted(list(set(files))) # Use list(set()) to ensure uniqueness and then sort

def format_file_size(bytes_size: int) -> str:
    """Format file size in bytes to human-readable format"""
    if bytes_size is None:
        return "0B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"

async def watch_directory(watch_path: Path, pipeline_instance: Pipeline, pattern_str: str, recursive: bool, auth_manager_instance: AuthManager): # Renamed parameters
    """Watch directory for new files"""
    if not WATCHDOG_AVAILABLE:
        console.print("[red]Watchdog library not installed. Cannot watch directory.[/red]")
        console.print("Install with: [cyan]pip install watchdog[/cyan]")
        return

    class VideoHandler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory:
                # Check if it matches our pattern
                for p_item in pattern_str.split(','): # Renamed pattern -> p_item
                    # A simple check, might need more robust glob-like matching for single files if patterns are complex
                    if event.src_path.endswith(p_item.strip().replace('*', '')): # Basic suffix check
                        asyncio.create_task( # Use asyncio.create_task for fire-and-forget
                            pipeline_instance.process_video(event.src_path, auth_manager_instance)
                        )
                        console.print(f"[green]New file queued: {event.src_path}[/green]")
                        break
    
    handler = VideoHandler()
    observer = Observer()
    observer.schedule(handler, str(watch_path), recursive=recursive)
    observer.start()
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    app()