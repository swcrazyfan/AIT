"""
Vector embeddings generation using BAAI/bge-m3 via DeepInfra.
Following Supabase best practices for hybrid search.
"""
import os
from typing import Tuple, Dict, Any, List, Optional # Added Optional
import structlog
import tiktoken
from openai import OpenAI # Ensure openai is in requirements.txt

# Assuming AuthManager is in core.auth
# This import is problematic if embeddings.py is a low-level utility.
# store_embeddings might be better placed in a service layer or within the step itself.
# For now, following the plan.
from video_tool.core.auth import AuthManager
# Assuming VideoIngestOutput or a similar structure for video_data in prepare_embedding_content
# This model is defined in enhanced_models.py in the plan.
# from video_tool.core.enhanced_models import EnhancedVideoIngestOutput
# For now, let's use a more generic type hint or assume the structure.

logger = structlog.get_logger(__name__)

def get_embedding_client() -> OpenAI: # Added return type hint
    """Get OpenAI client configured for DeepInfra API"""
    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        # It's better to raise an error if the key is essential
        logger.error("DEEPINFRA_API_KEY environment variable is not set.")
        raise ValueError("DEEPINFRA_API_KEY environment variable is required for embeddings.")
    
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai"
    )

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e: # Catching generic Exception as tiktoken might raise various errors
        logger.warning(f"Tiktoken encoding failed: {e}. Falling back to rough estimate.")
        # Fallback to rough estimate
        return len(text) // 4

def truncate_text(text: str, max_tokens: int = 3500) -> Tuple[str, str]:
    """Intelligently truncate text to fit token limit with sentence boundaries"""
    if not text: # Handle empty string case
        return "", "none"

    token_count = count_tokens(text)
    
    if token_count <= max_tokens:
        return text, "none"
    
    # Attempt sentence boundary truncation first
    try:
        sentences = text.split('. ')
        if len(sentences) > 1:
            rebuilt_text = ""
            for i, sentence in enumerate(sentences):
                # Append sentence and a period, unless it's the last part and might not be a full sentence
                current_segment = sentence + (". " if i < len(sentences) - 1 and sentences[i+1] else "")
                
                if count_tokens(rebuilt_text + current_segment) > max_tokens:
                    if rebuilt_text: # We have at least one complete sentence within limits
                        return rebuilt_text.rstrip(), "sentence_boundary"
                    else: # First sentence itself is too long
                        break # Fall through to token-based truncation
                rebuilt_text += current_segment
            
            # If loop finishes and rebuilt_text is still too long (e.g. single very long sentence)
            # or if it's empty (first sentence too long), fall through.
            if rebuilt_text and count_tokens(rebuilt_text) <= max_tokens:
                 return rebuilt_text.rstrip(), "sentence_boundary"


    except Exception as e:
        logger.warning(f"Sentence boundary truncation failed: {e}. Proceeding with token-based.")

    # Token-based truncation if sentence boundary fails or not applicable
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        truncated_tokens = tokens[:max_tokens]
        # Ensure decoding doesn't error out on partial tokens if possible
        truncated_text = encoding.decode(truncated_tokens, errors='ignore') 
        return truncated_text + "...", "token_boundary"
    except Exception as e:
        logger.warning(f"Token-based truncation failed: {e}. Falling back to character-based.")

    # Fallback: character-based truncation (less ideal)
    # Estimate characters per token (very rough)
    avg_chars_per_token = len(text) / token_count if token_count > 0 else 4
    char_limit = int(max_tokens * avg_chars_per_token)
    
    if len(text) <= char_limit: # Should have been caught by token_count check, but as a safeguard
        return text, "none"
    
    truncated_char_text = text[:char_limit]
    # Try to cut at last sentence boundary within character limit
    last_period_char = truncated_char_text.rfind('. ')
    if last_period_char > char_limit * 0.7:  # Only if we keep at least 70% of content
        return truncated_char_text[:last_period_char + 1], "character_sentence_boundary"
    
    return truncated_char_text + "...", "character_boundary"

def prepare_search_embeddings(query: str) -> Tuple[str, str]:
    """
    Prepare search query for embedding generation.
    
    Args:
        query: Search query text
        
    Returns:
        Tuple of (summary_content, keyword_content)
    """
    # For search queries, the plan uses the query for both summary and keyword content,
    # prefixing summary content.
    summary_content = f"Video content about: {query}" # As per plan
    keyword_content = query # As per plan
    
    return summary_content, keyword_content

# Placeholder for VideoIngestOutput structure or relevant parts
# This should align with what the AI analysis steps produce.
# For now, using Dict[str, Any] and checking for keys.
def prepare_embedding_content(video_data: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """
    Prepare semantic content for embedding generation optimized for hybrid search.
    
    Args:
        video_data: Dictionary representing video metadata and analysis results.
                    Expected to have keys like 'analysis', which in turn has 'ai_analysis',
                    'content_analysis', 'audio_analysis' etc.
                    This should ideally be a Pydantic model like EnhancedVideoIngestOutput.
        
    Returns:
        Tuple of (summary_content, keyword_content, metadata for truncation/tokens)
    """
    summary_parts = []
    keyword_concepts = []
    
    analysis = video_data.get('analysis', {})
    ai_analysis = analysis.get('ai_analysis', {}) # This is the ComprehensiveAIAnalysis in the plan
    
    # SUMMARY EMBEDDING: Semantic narrative content
    if ai_analysis:
        summary_obj = ai_analysis.get('summary', {}) # This is AIAnalysisSummary
        if summary_obj.get('content_category'):
            summary_parts.append(f"This is {summary_obj['content_category']} content")
        if summary_obj.get('overall'):
            summary_parts.append(summary_obj['overall'])
        if summary_obj.get('key_activities'):
            activities_text = ", ".join(summary_obj['key_activities'])
            summary_parts.append(f"Key activities include: {activities_text}")

        content_analysis_obj = ai_analysis.get('content_analysis', {}) # This is ContentAnalysis
        entities_obj = content_analysis_obj.get('entities', {}) # This is Entities
        if entities_obj and entities_obj.get('locations'):
            locations_text = []
            for loc in entities_obj['locations']:
                locations_text.append(f"{loc.get('name')} ({loc.get('type')})")
            if locations_text:
                summary_parts.append(f"Filmed at: {', '.join(locations_text)}")

    summary_content = ". ".join(filter(None, summary_parts))
    
    # KEYWORD EMBEDDING: Concept tags and semantic keywords
    if ai_analysis:
        audio_analysis_obj = ai_analysis.get('audio_analysis', {}) # This is AudioAnalysis
        transcript_obj = audio_analysis_obj.get('transcript', {}) # This is Transcript
        if transcript_obj and transcript_obj.get('full_text'):
            # Extract key phrases from transcript (first 200 chars for concepts as per plan)
            transcript_preview = transcript_obj['full_text'][:200]
            keyword_concepts.append(transcript_preview)

        content_analysis_obj = ai_analysis.get('content_analysis', {})
        entities_obj = content_analysis_obj.get('entities', {})
        visual_concepts = []
        if entities_obj:
            if entities_obj.get('locations'):
                for loc in entities_obj['locations']:
                    visual_concepts.extend(filter(None, [loc.get('name'), loc.get('type')]))
            if entities_obj.get('objects_of_interest'):
                for obj in entities_obj['objects_of_interest']:
                    visual_concepts.append(obj.get('object'))
        
        keyword_concepts.extend(filter(None, visual_concepts))

    keyword_content = " ".join(filter(None, keyword_concepts))
    
    # Truncate both contents
    final_summary_content, summary_truncation = truncate_text(summary_content, 3500)
    final_keyword_content, keyword_truncation = truncate_text(keyword_content, 3500)
    
    truncation_metadata = {
        'summary_token_count': count_tokens(final_summary_content),
        'keyword_token_count': count_tokens(final_keyword_content),
        'summary_truncation_method': summary_truncation, # Renamed for clarity
        'keyword_truncation_method': keyword_truncation, # Renamed for clarity
        'original_summary_token_count': count_tokens(summary_content), # Add original counts
        'original_keyword_token_count': count_tokens(keyword_content)
    }
    
    return final_summary_content, final_keyword_content, truncation_metadata

async def generate_embeddings(summary_content: str, keyword_content: str) -> Tuple[Optional[List[float]], Optional[List[float]]]: # Added Optional
    """Generate embeddings using BAAI/bge-m3 via DeepInfra"""
    if not summary_content and not keyword_content:
        logger.warning("Both summary and keyword content are empty. Skipping embedding generation.")
        return None, None
        
    try:
        client = get_embedding_client()
        
        summary_embedding: Optional[List[float]] = None
        if summary_content:
            summary_response = await asyncio.to_thread(
                client.embeddings.create,
                model="BAAI/bge-m3",
                input=summary_content
            )
            summary_embedding = summary_response.data[0].embedding
        
        keyword_embedding: Optional[List[float]] = None
        if keyword_content:
            keyword_response = await asyncio.to_thread(
                client.embeddings.create,
                model="BAAI/bge-m3", 
                input=keyword_content
            )
            keyword_embedding = keyword_response.data[0].embedding
        
        return summary_embedding, keyword_embedding
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}", exc_info=True)
        # Depending on policy, might want to return (None, None) or re-raise
        return None, None # Return None on failure to allow graceful handling

async def store_embeddings(
    clip_id: str, 
    summary_embedding: Optional[List[float]], 
    keyword_embedding: Optional[List[float]],
    summary_content: str, # The (potentially truncated) content that was embedded
    keyword_content: str, # The (potentially truncated) content that was embedded
    metadata: Dict[str, Any] # Contains token counts, truncation methods
) -> bool:
    """Store embeddings in Supabase database following pgvector patterns"""
    if summary_embedding is None and keyword_embedding is None:
        logger.warning(f"No embeddings provided to store for clip {clip_id}.")
        return False

    try:
        auth_manager = AuthManager() # This might need to be passed in or handled differently
                                     # if embeddings.py is a pure utility module.
        client = await auth_manager.get_authenticated_client() # Requires async AuthManager
        
        if not client:
            logger.error("Authentication required to store embeddings, but client is not available.")
            # raise ValueError("Authentication required to store embeddings")
            return False # Fail gracefully if no auth
        
        user_session = await auth_manager.get_current_session() # Requires async AuthManager
        if not user_session or 'user_id' not in user_session:
            logger.error("User session or user_id not found, cannot store embeddings.")
            return False
        user_id = user_session['user_id']
        
        vector_data = {
            'clip_id': clip_id,
            'user_id': user_id,
            'embedding_type': 'full_clip', # As per plan's schema for 'vectors' table
            'embedding_source': 'BAAI/bge-m3', # Model used
            'summary_vector': summary_embedding,
            'keyword_vector': keyword_embedding,
            'embedded_content': f"Summary: {summary_content}\nKeywords: {keyword_content}",
            # 'original_content' field in schema, might store pre-truncation text if different
            # For now, using embedded_content for original_content as well.
            'original_content': f"Summary: {summary_content}\nKeywords: {keyword_content}", 
            'token_count': metadata.get('summary_token_count', 0) + metadata.get('keyword_token_count', 0),
            'original_token_count': metadata.get('original_summary_token_count', 0) + \
                                    metadata.get('original_keyword_token_count', 0),
            'truncation_method': f"summary:{metadata.get('summary_truncation_method', 'none')}, keyword:{metadata.get('keyword_truncation_method', 'none')}"
        }
        
        # Filter out None vector fields before insertion if your DB schema requires non-null vectors
        # or if you want to insert only if at least one vector is present.
        # For now, assuming schema allows NULL for summary_vector/keyword_vector.
        
        result = await client.table('vectors').insert(vector_data).execute()
        
        # Check for errors in the response
        if hasattr(result, 'error') and result.error:
            logger.error(f"Failed to store embeddings for clip {clip_id}: {result.error.message}")
            return False
        elif not result.data: # Check if data is empty, indicating potential failure
            logger.error(f"Failed to store embeddings for clip {clip_id}, no data returned.", response=result)
            return False

        logger.info(f"Successfully stored embeddings for clip {clip_id}")
        return True
            
    except Exception as e:
        logger.error(f"Error storing embeddings for clip {clip_id}: {str(e)}", exc_info=True)
        return False

# Example of how video_data might look for prepare_embedding_content,
# based on EnhancedVideoIngestOutput and its nested models.
# This is for illustration and testing, not part of the module itself.
_example_video_data_for_prepare = {
    "analysis": {
        "ai_analysis": { # ComprehensiveAIAnalysis
            "summary": { # AIAnalysisSummary
                "content_category": "Tutorial",
                "overall": "A detailed tutorial on Python programming.",
                "key_activities": ["coding", "explaining concepts", "debugging"]
            },
            "content_analysis": { # ContentAnalysis
                "entities": { # Entities
                    "locations": [
                        {"name": "Home Office", "type": "Indoor"},
                        {"name": "Online Platform", "type": "Virtual"}
                    ],
                    "objects_of_interest": [
                        {"object": "Laptop"}, {"object": "Code Editor"}
                    ]
                }
            },
            "audio_analysis": { # AudioAnalysis
                "transcript": { # Transcript
                    "full_text": "Hello everyone, today we are going to learn Python. Python is a versatile language..."
                }
            }
        }
    }
}