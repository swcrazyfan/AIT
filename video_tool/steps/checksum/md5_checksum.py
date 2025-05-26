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
    provides = ["file_checksum", "file_size_bytes", "processing_status"] # processing_status here refers to the JSONB column

    async def process(self, video: VideoMetadata, context: dict = None) -> StepResult: # Added context
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
                "file_checksum": checksum,
                "file_size_bytes": file_size,
            }

            # This dictionary is for the JSONB 'processing_status' column in 'clips'
            jsonb_status_update = {
                "duplicate_info": {
                    "is_duplicate": is_duplicate,
                    "checksum_algorithm": "md5",
                    "checked_at": datetime.now().isoformat() # Changed to datetime.now() for consistency
                }
            }

            if is_duplicate:
                # This special key signals BaseStep to update the main TEXT processing_status column
                jsonb_status_update["_overall_status_enum_override_"] = ProcessingStatus.DUPLICATE.value
                jsonb_status_update["duplicate_info"]["duplicate_of"] = duplicate_check.data[0]['id']
                jsonb_status_update["duplicate_info"]["duplicate_file_name"] = duplicate_check.data[0]['file_name']
            
            result_data["processing_status"] = jsonb_status_update # Key matches JSONB column name

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