"""
Comprehensive search system with semantic, full-text, and hybrid search capabilities.
"""
import asyncio
from typing import Dict, List, Optional, Tuple, Literal, Any
from datetime import datetime
import structlog

from video_tool.core.db import get_supabase
from video_tool.core.auth import AuthManager
from video_tool.embeddings import prepare_search_embeddings, generate_embeddings # Updated import

logger = structlog.get_logger(__name__)

SearchType = Literal["semantic", "fulltext", "hybrid", "transcripts", "similar"]

class VideoSearcher:
    """Comprehensive video search with multiple search strategies"""

    def __init__(self):
        self.auth_manager = AuthManager()

    async def search(
        self,
        query: str,
        search_type: SearchType = "hybrid",
        match_count: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform comprehensive search across video catalog.

        Args:
            query: Search query text
            search_type: Type of search (semantic, fulltext, hybrid, transcripts, similar)
            match_count: Number of results to return
            filters: Optional filters (camera_make, content_category, etc.)
            weights: Search weights for hybrid search

        Returns:
            List of matching video clips with metadata
        """
        client = await self._get_authenticated_client()
        user_id = await self._get_current_user_id() # Await the async call

        if not client or not user_id:
            raise ValueError("Authentication required for search")

        # Apply search strategy
        if search_type == "semantic":
            return await self._semantic_search(client, user_id, query, match_count, weights)
        elif search_type == "fulltext":
            return await self._fulltext_search(client, user_id, query, match_count)
        elif search_type == "hybrid":
            return await self._hybrid_search(client, user_id, query, match_count, weights)
        elif search_type == "transcripts":
            return await self._transcript_search(client, user_id, query, match_count)
        else:
            raise ValueError(f"Unknown search type: {search_type}")

    async def find_similar(
        self,
        clip_id: str,
        match_count: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find clips similar to a given clip using vector similarity.

        Args:
            clip_id: ID of the source clip
            match_count: Number of similar clips to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of similar clips
        """
        client = await self._get_authenticated_client()
        user_id = await self._get_current_user_id() # Await the async call

        try:
            result = await client.rpc('find_similar_clips', {
                'source_clip_id': clip_id,
                'user_id_filter': user_id,
                'match_count': match_count,
                'similarity_threshold': similarity_threshold
            }).execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Similar search failed: {str(e)}")
            return []

    async def _semantic_search(
        self,
        client,
        user_id: str,
        query: str,
        match_count: int,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using vector embeddings"""
        try:
            # Prepare content for embedding
            summary_content, keyword_content = prepare_search_embeddings(query)

            # Generate embeddings
            summary_embedding, keyword_embedding = await generate_embeddings(
                summary_content,
                keyword_content
            )

            # Set default weights
            search_params = {
                'query_summary_embedding': summary_embedding,
                'query_keyword_embedding': keyword_embedding,
                'user_id_filter': user_id,
                'match_count': match_count,
                'summary_weight': weights.get('summary', 1.0) if weights else 1.0,
                'keyword_weight': weights.get('keyword', 0.8) if weights else 0.8,
                'similarity_threshold': weights.get('threshold', 0.0) if weights else 0.0
            }

            # Execute semantic search
            result = await client.rpc('semantic_search_clips', search_params).execute()
            return result.data or []

        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []

    async def _fulltext_search(
        self,
        client,
        user_id: str,
        query: str,
        match_count: int
    ) -> List[Dict[str, Any]]:
        """Perform full-text search"""
        try:
            result = await client.rpc('fulltext_search_clips', {
                'query_text': query,
                'user_id_filter': user_id,
                'match_count': match_count
            }).execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Full-text search failed: {str(e)}")
            return []

    async def _hybrid_search(
        self,
        client,
        user_id: str,
        query: str,
        match_count: int,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining full-text and semantic search"""
        try:
            # Prepare embeddings
            summary_content, keyword_content = prepare_search_embeddings(query)
            summary_embedding, keyword_embedding = await generate_embeddings(
                summary_content,
                keyword_content
            )

            # Set default weights for RRF
            default_weights = {
                'fulltext': 1.0,
                'summary': 1.0,
                'keyword': 0.8,
                'rrf_k': 50
            }

            if weights:
                default_weights.update(weights)

            # Execute hybrid search
            result = await client.rpc('hybrid_search_clips', {
                'query_text': query,
                'query_summary_embedding': summary_embedding,
                'query_keyword_embedding': keyword_embedding,
                'user_id_filter': user_id,
                'match_count': match_count,
                'fulltext_weight': default_weights['fulltext'],
                'summary_weight': default_weights['summary'],
                'keyword_weight': default_weights['keyword'],
                'rrf_k': default_weights['rrf_k']
            }).execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            return []

    async def _transcript_search(
        self,
        client,
        user_id: str,
        query: str,
        match_count: int
    ) -> List[Dict[str, Any]]:
        """Perform search specifically on transcripts"""
        try:
            result = await client.rpc('search_transcripts', {
                'query_text': query,
                'user_id_filter': user_id,
                'match_count': match_count,
                'min_content_length': 50
            }).execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Transcript search failed: {str(e)}")
            return []

    async def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics for the video catalog"""
        client = await self._get_authenticated_client()

        if not client:
            return {}

        try:
            result = await client.rpc('get_user_stats').execute()
            return result.data or {}

        except Exception as e:
            logger.error(f"Failed to get user stats: {str(e)}")
            return {}

    async def get_clip_details(self, clip_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific clip"""
        client = await self._get_authenticated_client()
        user_id = await self._get_current_user_id() # Await the async call

        if not client or not user_id:
            return None

        try:
            # Get clip details
            clip_result = await client.table('clips')\
                .select('*')\
                .eq('id', clip_id)\
                .eq('user_id', user_id)\
                .single()\
                .execute()

            if not clip_result.data:
                return None

            clip_data = clip_result.data

            # Get transcript if available
            transcript_result = await client.table('transcripts')\
                .select('full_text')\
                .eq('clip_id', clip_id)\
                .eq('user_id', user_id)\
                .execute()

            if transcript_result.data:
                clip_data['transcript'] = transcript_result.data[0]['full_text']

            # Get AI analysis if available
            analysis_result = await client.table('analysis')\
                .select('*')\
                .eq('clip_id', clip_id)\
                .eq('user_id', user_id)\
                .execute()

            if analysis_result.data:
                clip_data['ai_analysis'] = analysis_result.data

            return clip_data

        except Exception as e:
            logger.error(f"Failed to get clip details: {str(e)}")
            return None

    async def _get_authenticated_client(self):
        """Get authenticated Supabase client"""
        return await self.auth_manager.get_authenticated_client()

    async def _get_current_user_id(self) -> Optional[str]: # Make async
        """Get current authenticated user ID"""
        session = await self.auth_manager.get_current_session() # Await the async call
        return session['user_id'] if session else None

def format_search_results(
    results: List[Dict[str, Any]],
    search_type: str,
    show_scores: bool = False
) -> List[Dict[str, Any]]:
    """
    Format search results for display.

    Args:
        results: Raw search results
        search_type: Type of search performed
        show_scores: Whether to include similarity/ranking scores

    Returns:
        Formatted results for display
    """
    formatted_results = []

    for result in results:
        formatted = {
            'id': result.get('id'),
            'file_name': result.get('file_name'),
            'content_summary': result.get('content_summary'),
            'content_category': result.get('content_category'),
            'duration': format_duration(result.get('duration_seconds', 0)),
            'camera': f"{result.get('camera_make', '')} {result.get('camera_model', '')}".strip(),
            'processed_at': result.get('processed_at', ''),
            'tags': result.get('content_tags', [])
        }

        # Add search-specific fields
        if show_scores:
            if search_type == "semantic":
                formatted['similarity_score'] = f"{result.get('combined_similarity', 0):.3f}"
            elif search_type == "hybrid":
                formatted['search_rank'] = f"{result.get('search_rank', 0):.3f}"
                formatted['match_type'] = result.get('match_type', 'unknown')
            elif search_type == "fulltext":
                formatted['fts_rank'] = f"{result.get('fts_rank', 0):.3f}"

        formatted_results.append(formatted)

    return formatted_results

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"