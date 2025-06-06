from typing import Dict, Any, List, Optional, Tuple
import structlog

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepResult, VideoMetadata, StepConfig
# Import the centralized embedding utilities
from video_tool.embeddings import (
    prepare_embedding_content,
    generate_embeddings,
    store_embeddings, # May or may not be called directly by the step, depends on design
    count_tokens # For providing metadata if needed
)

logger = structlog.get_logger(__name__)

class BgeEmbeddingsStep(BaseStep):
    """
    Generates and stores BAAI/bge-m3 embeddings for video content.
    Uses centralized embedding utilities.
    """
    
    name = "bge_embeddings"
    version = "1.0"
    description = "Generate BAAI/bge-m3 embeddings for video summary and keywords."
    category = "embeddings"
    
    # Requires data typically generated by AI analysis steps
    requires = ["video_id", "user_id"] # user_id for store_embeddings
    # It implicitly requires fields used by prepare_embedding_content from VideoMetadata,
    # such as video.analysis.ai_analysis.summary etc.
    # These are not explicitly listed as 'requires' here because they are deeply nested.
    # The prepare_embedding_content function handles missing nested data gracefully.
    
    provides = ["embeddings_stored"] # Indicates if embeddings were successfully stored

    def __init__(self, config: StepConfig):
        super().__init__(config)
        self.api_key_env_var = "DEEPINFRA_API_KEY" # As used in video_tool.embeddings
        self._check_api_key()

    def _check_api_key(self):
        if not os.getenv(self.api_key_env_var):
            self.logger.warning(f"{self.api_key_env_var} not set. {self.name} step might fail if API call is made.")
            # The actual error will be raised by get_embedding_client if called without key.

    async def process(self, video: VideoMetadata, context: Optional[Dict[str, Any]] = None) -> StepResult:
        self.logger.info(f"Starting BGE embeddings generation for video_id: {video.video_id}")

        if not video.user_id:
            msg = "user_id is missing in VideoMetadata, cannot store embeddings."
            self.logger.error(msg, video_id=video.video_id)
            return StepResult(success=False, step_name=self.name, video_id=video.video_id, error=msg)

        try:
            # 1. Prepare content for embeddings
            # prepare_embedding_content expects a dict-like structure or an object with attributes.
            # VideoMetadata is a Pydantic model, so model_dump() is appropriate.
            video_data_dict = video.model_dump(exclude_none=True) 
            
            summary_content_str, keyword_content_str, prep_metadata = prepare_embedding_content(video_data_dict)
            
            if not summary_content_str and not keyword_content_str:
                msg = "No content available to generate embeddings."
                self.logger.warning(msg, video_id=video.video_id)
                return StepResult(
                    success=True, # Success because there's nothing to do
                    step_name=self.name, 
                    video_id=video.video_id, 
                    data={"embeddings_stored": False, "reason": msg},
                    metadata=prep_metadata
                )

            self.logger.debug("Prepared content for BGE embeddings", video_id=video.video_id, 
                              summary_len=len(summary_content_str), keyword_len=len(keyword_content_str),
                              prep_meta=prep_metadata)

            # 2. Generate embeddings
            summary_embedding, keyword_embedding = await generate_embeddings(
                summary_content_str, 
                keyword_content_str
            )

            if summary_embedding is None and keyword_embedding is None:
                # generate_embeddings logs its own errors
                msg = "Failed to generate any embeddings."
                self.logger.error(msg, video_id=video.video_id)
                return StepResult(success=False, step_name=self.name, video_id=video.video_id, error=msg, metadata=prep_metadata)

            self.logger.info("Embeddings generated successfully.", video_id=video.video_id,
                             has_summary_emb=summary_embedding is not None,
                             has_keyword_emb=keyword_embedding is not None)
            
            # 3. Store embeddings
            # The store_embeddings function from video_tool.embeddings handles Supabase interaction.
            # It requires auth, which it tries to manage internally.
            # This step primarily orchestrates the call.
            storage_success = await store_embeddings(
                clip_id=video.video_id,
                summary_embedding=summary_embedding,
                keyword_embedding=keyword_embedding,
                summary_content=summary_content_str, # Pass the (potentially truncated) text
                keyword_content=keyword_content_str, # Pass the (potentially truncated) text
                metadata=prep_metadata # Pass token counts and truncation info
            )

            if not storage_success:
                # store_embeddings logs its own errors
                msg = "Failed to store generated embeddings."
                self.logger.error(msg, video_id=video.video_id)
                return StepResult(success=False, step_name=self.name, video_id=video.video_id, error=msg, metadata=prep_metadata)

            self.logger.info(f"BGE embeddings stored successfully for video_id: {video.video_id}")
            
            return StepResult(
                success=True,
                step_name=self.name,
                video_id=video.video_id,
                data={"embeddings_stored": True}, # Simple flag indicating success
                metadata={
                    "summary_embedding_size": len(summary_embedding) if summary_embedding else 0,
                    "keyword_embedding_size": len(keyword_embedding) if keyword_embedding else 0,
                    **prep_metadata # Include token counts and truncation info
                }
            )

        except Exception as e:
            self.logger.error(f"Error in BGE Embeddings step for video {video.video_id}: {str(e)}", exc_info=True)
            return StepResult(
                success=False,
                step_name=self.name,
                video_id=video.video_id,
                error=str(e)
            )

    async def setup(self):
        self.logger.info(f"{self.name} step initialized. Uses DeepInfra for BAAI/bge-m3 embeddings.")
        self._check_api_key() # Re-check or log status on setup

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "embedding_model": "BAAI/bge-m3",
            "embedding_service": "DeepInfra",
            "api_key_env_var": self.api_key_env_var,
            "api_key_set": bool(os.getenv(self.api_key_env_var))
        })
        return info

# Need to import os for getenv in __init__ and get_info
import os
