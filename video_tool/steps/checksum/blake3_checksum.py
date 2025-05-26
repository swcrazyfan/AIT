from pathlib import Path
import blake3 # Requires 'pip install blake3'

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata, ProcessingStatus

class Blake3ChecksumStep(BaseStep):
    """Fast BLAKE3 checksum calculation with duplicate detection"""

    name = "blake3_checksum"
    version = "1.0"
    description = "Calculate BLAKE3 checksum for deduplication (faster than MD5)"
    category = "checksum"

    requires = ["file_path"]
    provides = ["file_checksum", "file_size_bytes", "processing_status"] # processing_status here refers to the JSONB column

    async def process(self, video: VideoMetadata, context: dict = None) -> StepResult:
        try:
            file_path_str = video.file_path
            self.logger.info(f"Blake3ChecksumStep processing file: {file_path_str}", video_id=video.video_id) # Log input path
            file_path = Path(file_path_str)

            if not file_path.exists():
                return StepResult(
                    success=False,
                    step_name=self.name,
                    video_id=video.video_id,
                    error=f"File not found: {file_path}"
                )

            # Calculate BLAKE3
            blake3_hash = blake3.blake3()
            file_size = 0
            
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""): # Standard 64KB chunks
                    blake3_hash.update(chunk)
                    file_size += len(chunk)
            
            checksum = blake3_hash.hexdigest()
            self.logger.info(f"Calculated checksum: {checksum}, size: {file_size} for {file_path_str}", video_id=video.video_id) # Log calculated values

            # Ensure supabase client is initialized (it's done in BaseStep.execute)
            # but good practice to call ensure_supabase_client if directly using self.supabase
            await self._ensure_supabase_client()

            # Check for duplicates IMMEDIATELY
            # Note: The table 'clips' and its columns 'id', 'file_name', 'checksum' must exist.
            # user_id is also needed for RLS.
            duplicate_check_query = self.supabase.table('clips')\
                .select('id, file_name')\
                .eq('checksum', checksum)\
                .neq('id', video.video_id)
            
            if video.user_id: # Add user_id filter if available for RLS
                 duplicate_check_query = duplicate_check_query.eq('user_id', video.user_id)
            
            duplicate_check = await duplicate_check_query.execute()
            
            is_duplicate = len(duplicate_check.data) > 0

            self.logger.info(f"Calculated BLAKE3 checksum: {checksum}, duplicate: {is_duplicate}", video_id=video.video_id)

            result_data = {
                "file_checksum": checksum,
                "file_size_bytes": file_size,
            }

            # This dictionary is for the JSONB 'processing_status' column in 'clips'
            jsonb_status_update = {
                "duplicate_info": {
                    "is_duplicate": is_duplicate,
                    "checksum_algorithm": "blake3", # Specify algorithm
                    "checked_at": datetime.now().isoformat()
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
            self.logger.error(f"BLAKE3 checksum step failed: {str(e)}", video_id=video.video_id, exc_info=True)
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video.video_id,
                error=str(e)
            )