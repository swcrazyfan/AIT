import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import structlog

# Corrected import path for EnhancedVideoIngestOutput
from video_tool.core.enhanced_models import EnhancedVideoIngestOutput

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
            # Assuming the project root is two levels up from core directory
            project_root = Path(__file__).parent.parent.parent
            self.base_output_dir = project_root / "output"

        self.current_run_dir: Optional[Path] = None
        self.current_run_id: Optional[str] = None
        self.run_paths: Dict[str, Path] = {}

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
            "logs",           # Run-specific logs
            "artifacts",      # Step artifacts and intermediate files
            "reports",        # Summary reports and statistics
            "exports"         # Export-ready files
        ]

        self.current_run_dir.mkdir(parents=True, exist_ok=True) # Ensure base run_dir exists
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
        if not self.current_run_dir or not self.current_run_id:
            raise ValueError("Current run directory or ID not set")

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
        if not self.current_run_dir or not self.run_paths:
            raise ValueError("No active run. Call create_new_run() first.")

        saved_paths: Dict[str, Any] = {} # Use Any for list of thumbnails

        # 1. Save individual video JSON
        # Use video_output.id if available, otherwise generate one or use checksum
        video_identifier = video_output.id if hasattr(video_output, 'id') and video_output.id else video_output.file_info.file_checksum
        json_filename = f"{video_output.file_info.file_name}_{video_identifier}.json"
        json_path = self.run_paths["json"] / json_filename

        with open(json_path, 'w') as f:
            # Pydantic's model_dump or dict might be better if available on EnhancedVideoIngestOutput
            json.dump(video_output.model_dump() if hasattr(video_output, 'model_dump') else video_output.dict(), f, indent=2, default=str)
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
            for i, thumbnail_path_str in enumerate(video_output.thumbnails):
                thumbnail_path = Path(thumbnail_path_str)
                if thumbnail_path.exists():
                    # Create organized filename
                    thumb_filename = f"thumb_{i:03d}_{checksum}{thumbnail_path.suffix}" # Preserve original suffix
                    organized_path = thumbnail_dir / thumb_filename

                    # Copy to organized location
                    shutil.copy2(thumbnail_path, organized_path)
                    organized_thumbnails.append(str(organized_path))

            saved_paths["thumbnails"] = organized_thumbnails

        # 4. Save detailed AI analysis if available
        if (video_output.analysis and video_output.analysis.ai_analysis and
            video_output.analysis.ai_analysis.analysis_file_path):

            ai_analysis_file_path = Path(video_output.analysis.ai_analysis.analysis_file_path)
            if ai_analysis_file_path.exists():
                ai_analysis_name = f"{video_output.file_info.file_name}_AI_analysis{ai_analysis_file_path.suffix}"
                ai_analysis_path_dest = self.run_paths["ai_analysis"] / ai_analysis_name
                shutil.copy2(ai_analysis_file_path, ai_analysis_path_dest)
                saved_paths["ai_analysis"] = str(ai_analysis_path_dest)

        # 5. Update run statistics
        self._update_run_stats(video_output)

        logger.info(f"Saved organized output for {video_output.file_info.file_name}")
        return saved_paths # type: ignore

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
        if not self.current_run_dir or not self.current_run_id or not self.run_paths:
            raise ValueError("No active run.")

        # Create comprehensive summary
        summary = {
            "run_info": {
                "run_id": self.current_run_id,
                "created_at": datetime.now().isoformat(), # Should be run creation time from manifest
                "total_videos": len(videos),
                "successful_videos": len([v for v in videos if v.processing_status == "completed"]), # Assuming status field
                "failed_videos": len([v for v in videos if v.processing_status == "failed"]) # Assuming status field
            },
            "processing_stats": processing_stats,
            "videos": [v.model_dump() if hasattr(v, 'model_dump') else v.dict() for v in videos]
        }
        
        run_info_from_manifest = self.get_run_info()
        if run_info_from_manifest:
            summary["run_info"]["created_at"] = run_info_from_manifest.get("created_at", datetime.now().isoformat())


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
        
        # Update manifest with pipeline config info
        manifest_path = self.current_run_dir / "run_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r+') as f:
                manifest_data = json.load(f)
                manifest_data["pipeline_config"] = {"name": config.get("name", "unknown"), "version": config.get("version", "unknown")}
                f.seek(0)
                json.dump(manifest_data, f, indent=2)
                f.truncate()

        return str(config_path)

    def generate_processing_report(self) -> str:
        """Generate comprehensive processing report"""
        if not self.current_run_dir or not self.run_paths:
            raise ValueError("No active run.")

        # Read manifest for stats
        manifest = self.get_run_info()
        if not manifest:
            manifest = {"error": "Manifest not found"}


        # Collect individual video results
        json_files = list(self.run_paths["json"].glob("*.json"))
        video_results = []

        for json_file in json_files:
            with open(json_file, 'r') as f:
                try:
                    video_data = json.load(f)
                    video_results.append(video_data)
                except json.JSONDecodeError:
                    logger.error(f"Could not decode JSON from {json_file}")


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
        if not self.current_run_dir:
            logger.warning("Cannot update run stats, no current run directory.")
            return
        manifest_path = self.current_run_dir / "run_manifest.json"
        
        if not manifest_path.exists():
            logger.warning(f"Manifest file not found at {manifest_path}, cannot update stats.")
            return

        with open(manifest_path, 'r+') as f:
            manifest = json.load(f)

            # Update counts
            manifest["videos_processed"] = manifest.get("videos_processed", 0) + 1
            
            # Safely access duration from video_output
            duration = 0.0
            if video_output.video and hasattr(video_output.video, 'frame_rate') and video_output.video.frame_rate: # This is incorrect, should be duration_seconds
                 # The plan has `video_output.video.frame_rate` but it should be `video_output.video.duration_seconds` or similar
                 # For now, let's assume a duration field exists or default to 0
                 if hasattr(video_output.video, 'duration_seconds') and video_output.video.duration_seconds is not None:
                     duration = float(video_output.video.duration_seconds)
                 elif hasattr(video_output, 'duration_seconds') and video_output.duration_seconds is not None: # Fallback if it's on top level
                     duration = float(video_output.duration_seconds)


            manifest["total_duration_seconds"] = manifest.get("total_duration_seconds", 0.0) + duration
            manifest["total_size_bytes"] = manifest.get("total_size_bytes", 0) + video_output.file_info.file_size_bytes

            if video_output.processing_status == "failed": # Assuming status field
                 manifest["videos_failed"] = manifest.get("videos_failed", 0) + 1


            # Update status
            manifest["last_updated"] = datetime.now().isoformat()
            
            f.seek(0)
            json.dump(manifest, f, indent=2)
            f.truncate()

    def _generate_video_statistics(self, video_results: List[Dict]) -> Dict[str, Any]:
        """Generate video statistics for report"""
        if not video_results:
            return {}

        total_duration_s = 0
        for v_data in video_results:
            video_details = v_data.get("video", {})
            # Try to get duration from a few possible places
            duration = video_details.get("duration_seconds") # From EnhancedVideoIngestOutput.video.duration_seconds (not in plan but logical)
            if duration is None:
                # Fallback if EnhancedVideoIngestOutput.video.codec.duration_seconds exists (unlikely)
                codec_details = video_details.get("codec", {})
                duration = codec_details.get("duration_seconds")
            if duration is None:
                 # Fallback if EnhancedVideoIngestOutput.duration_seconds exists (top level)
                duration = v_data.get("duration_seconds")

            if duration is not None:
                try:
                    total_duration_s += float(duration)
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse duration: {duration} for a video.")


        stats = {
            "total_videos": len(video_results),
            "total_duration_hours": total_duration_s / 3600,
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
                camera = f"{camera_info['make']} {camera_info.get('model', '')}".strip()
                stats["camera_distribution"][camera] = stats["camera_distribution"].get(camera, 0) + 1

            video_info = video.get("video", {})
            resolution = video_info.get("resolution", {})
            if resolution.get("width") and resolution.get("height"):
                res_key = f"{resolution['width']}x{resolution['height']}"
                stats["resolution_distribution"][res_key] = stats["resolution_distribution"].get(res_key, 0) + 1

            codec_details = video_info.get("codec", {})
            codec_name = codec_details.get("name")
            if codec_name:
                stats["codec_distribution"][codec_name] = stats["codec_distribution"].get(codec_name, 0) + 1
        
        return stats

    def _generate_processing_timeline(self, video_results: List[Dict]) -> List[Dict[str, Any]]:
        """Generate processing timeline"""
        timeline = []

        for video in video_results:
            file_info = video.get("file_info", {})
            video_details = video.get("video", {})
            duration = video_details.get("duration_seconds") # from EnhancedVideoIngestOutput.video.duration_seconds
            if duration is None:
                duration = video.get("duration_seconds") # Fallback

            timeline.append({
                "file_name": file_info.get("file_name"),
                "processed_at": file_info.get("processed_at"),
                "processing_status": video.get("processing_status"),
                "duration_seconds": duration
            })

        # Sort by processing time
        timeline.sort(key=lambda x: x.get("processed_at") or "")
        return timeline

    def _generate_quality_analysis(self, video_results: List[Dict]) -> Dict[str, Any]:
        """Generate quality analysis summary"""
        quality_stats = {
            "hdr_videos": 0,
            "4k_videos": 0, # Assuming 4K means height >= 2160
            "exposure_warnings": 0,
            # "average_technical_quality": {} # This was in plan but not implemented, needs definition
        }

        for video in video_results:
            video_info = video.get("video", {})
            # HDR detection
            color_info = video_info.get("color", {}).get("hdr", {})
            if color_info.get("is_hdr"):
                quality_stats["hdr_videos"] += 1

            # 4K detection
            resolution = video_info.get("resolution", {})
            if resolution.get("height", 0) >= 2160:
                quality_stats["4k_videos"] += 1

            # Exposure warnings
            exposure = video_info.get("exposure", {})
            if exposure.get("warning"):
                quality_stats["exposure_warnings"] += 1

        return quality_stats

    def _generate_technical_summary(self, video_results: List[Dict]) -> Dict[str, Any]:
        """Generate technical summary"""
        num_videos = len(video_results)
        if num_videos == 0:
            return {
                "total_files_processed": 0,
                "average_file_size_mb": 0,
                "processing_completion_rate": 0
            }

        total_size_bytes = sum(v.get("file_info", {}).get("file_size_bytes", 0) for v in video_results)
        completed_videos = len([v for v in video_results if v.get("processing_status") == "completed"])

        return {
            "total_files_processed": num_videos,
            "average_file_size_mb": (total_size_bytes / num_videos) / (1024**2) if num_videos > 0 else 0,
            "processing_completion_rate": (completed_videos / num_videos) if num_videos > 0 else 0
        }

    def get_run_info(self) -> Optional[Dict[str, Any]]:
        """Get current run information"""
        if not self.current_run_dir:
            return None

        manifest_path = self.current_run_dir / "run_manifest.json"
        if not manifest_path.exists():
            return None

        with open(manifest_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Could not decode manifest JSON from {manifest_path}")
                return None


    def list_available_runs(self) -> List[Dict[str, Any]]:
        """List all available runs"""
        runs = []
        if not self.runs_dir.exists():
            return runs

        for run_dir_item in self.runs_dir.iterdir():
            if run_dir_item.is_dir():
                manifest_path = run_dir_item / "run_manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        try:
                            manifest = json.load(f)
                            runs.append({
                                "run_id": manifest.get("run_id"),
                                "created_at": manifest.get("created_at"),
                                "status": manifest.get("status"),
                                "videos_processed": manifest.get("videos_processed", 0),
                                "path": str(run_dir_item)
                            })
                        except json.JSONDecodeError:
                             logger.error(f"Could not decode manifest JSON from {manifest_path} for run {run_dir_item.name}")


        # Sort by creation time (newest first)
        runs.sort(key=lambda x: x.get("created_at") or "", reverse=True)
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
                try:
                    shutil.rmtree(run_path)
                    logger.info(f"Cleaned up old run: {run_info['run_id']}")
                except OSError as e:
                    logger.error(f"Error removing old run {run_info['run_id']} at {run_path}: {e}")


# Global output manager instance
output_manager = OutputManager()

def get_output_manager() -> OutputManager:
    """Get the global output manager instance"""
    return output_manager