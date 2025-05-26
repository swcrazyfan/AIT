-- =====================================================
-- HYBRID SEARCH FUNCTIONS FOR VIDEO CATALOG
-- =====================================================

-- Function for basic semantic search using vector similarity
CREATE OR REPLACE FUNCTION public.semantic_search_clips(
  p_query_summary_embedding vector(1024),
  p_query_keyword_embedding vector(1024),
  p_user_id_filter UUID,
  p_match_count INT DEFAULT 10,
  p_summary_weight FLOAT DEFAULT 1.0,
  p_keyword_weight FLOAT DEFAULT 0.8,
  p_similarity_threshold FLOAT DEFAULT 0.0
)
RETURNS TABLE (
  id UUID,
  file_name TEXT,
  local_path TEXT,
  content_summary TEXT,
  content_tags TEXT[],
  duration_seconds NUMERIC,
  camera_make TEXT,
  camera_model TEXT,
  content_category TEXT,
  processed_at TIMESTAMPTZ,
  summary_similarity FLOAT,
  keyword_similarity FLOAT,
  combined_similarity FLOAT
)
LANGUAGE SQL
SECURITY INVOKER -- Can be invoker as RLS on underlying tables will apply
SET search_path = ''
AS $$
WITH summary_search AS (
  SELECT
    c.id, c.file_name, c.local_path, c.content_summary, 
    c.content_tags, c.duration_seconds, c.camera_make, c.camera_model,
    c.content_category, c.processed_at,
    (v.summary_vector <#> p_query_summary_embedding) * -1 as summary_similarity, -- Cosine distance is <#>, similarity is 1 - distance or inner product
    ROW_NUMBER() OVER (ORDER BY (v.summary_vector <#> p_query_summary_embedding)) as rank_ix -- Order by distance for similarity
  FROM public.clips c
  JOIN public.vectors v ON c.id = v.clip_id
  WHERE c.user_id = p_user_id_filter
    AND v.embedding_type = 'full_clip'
    AND v.summary_vector IS NOT NULL
  ORDER BY (v.summary_vector <#> p_query_summary_embedding) -- Smallest distance first
  LIMIT LEAST(p_match_count * 2, 50)
),
keyword_search AS (
  SELECT
    c.id,
    (v.keyword_vector <#> p_query_keyword_embedding) * -1 as keyword_similarity
  FROM public.clips c
  JOIN public.vectors v ON c.id = v.clip_id
  WHERE c.user_id = p_user_id_filter
    AND v.embedding_type = 'full_clip'
    AND v.keyword_vector IS NOT NULL
  ORDER BY (v.keyword_vector <#> p_query_keyword_embedding)
  LIMIT LEAST(p_match_count * 2, 50)
)
SELECT
  ss.id,
  ss.file_name,
  ss.local_path,
  ss.content_summary,
  ss.content_tags,
  ss.duration_seconds,
  ss.camera_make,
  ss.camera_model,
  ss.content_category,
  ss.processed_at,
  ss.summary_similarity,
  COALESCE(ks.keyword_similarity, 0.0) as keyword_similarity,
  (ss.summary_similarity * p_summary_weight + COALESCE(ks.keyword_similarity, 0.0) * p_keyword_weight) as combined_similarity
FROM summary_search ss
LEFT JOIN keyword_search ks ON ss.id = ks.id
WHERE ss.summary_similarity >= p_similarity_threshold -- Ensure this is meaningful for inner product
ORDER BY combined_similarity DESC
LIMIT p_match_count;
$$;

-- Function for hybrid search combining full-text and semantic search using RRF
CREATE OR REPLACE FUNCTION public.hybrid_search_clips(
  p_query_text TEXT,
  p_query_summary_embedding vector(1024),
  p_query_keyword_embedding vector(1024),
  p_user_id_filter UUID,
  p_match_count INT DEFAULT 10,
  p_fulltext_weight FLOAT DEFAULT 1.0,
  p_summary_weight FLOAT DEFAULT 1.0,
  p_keyword_weight FLOAT DEFAULT 0.8,
  p_rrf_k INT DEFAULT 50
)
RETURNS TABLE (
  id UUID,
  file_name TEXT,
  local_path TEXT,
  content_summary TEXT,
  content_tags TEXT[],
  duration_seconds NUMERIC,
  camera_make TEXT,
  camera_model TEXT,
  content_category TEXT,
  processed_at TIMESTAMPTZ,
  transcript_preview TEXT,
  similarity_score FLOAT, -- Representing semantic similarity part
  search_rank FLOAT,      -- RRF combined rank
  match_type TEXT
)
LANGUAGE SQL
SECURITY INVOKER
SET search_path = ''
AS $$
WITH fulltext_search AS ( -- Renamed for clarity
  SELECT
    c.id, c.file_name, c.local_path, c.content_summary, 
    c.content_tags, c.duration_seconds, c.camera_make, c.camera_model,
    c.content_category, c.processed_at, c.transcript_preview,
    ts_rank_cd(c.fts, websearch_to_tsquery('english', p_query_text)) as fts_score,
    ROW_NUMBER() OVER(ORDER BY ts_rank_cd(c.fts, websearch_to_tsquery('english', p_query_text)) DESC) as rank_ix
  FROM public.clips c
  WHERE c.user_id = p_user_id_filter
    AND c.fts @@ websearch_to_tsquery('english', p_query_text)
  ORDER BY fts_score DESC
  LIMIT LEAST(p_match_count * 2, 30)
),
summary_semantic_search AS ( -- Renamed for clarity
  SELECT
    c.id, c.file_name, c.local_path, c.content_summary,
    c.content_tags, c.duration_seconds, c.camera_make, c.camera_model,
    c.content_category, c.processed_at, c.transcript_preview,
    (v.summary_vector <#> p_query_summary_embedding) * -1 as similarity_score,
    ROW_NUMBER() OVER (ORDER BY (v.summary_vector <#> p_query_summary_embedding)) as rank_ix
  FROM public.clips c
  JOIN public.vectors v ON c.id = v.clip_id
  WHERE c.user_id = p_user_id_filter
    AND v.embedding_type = 'full_clip'
    AND v.summary_vector IS NOT NULL
  ORDER BY rank_ix ASC
  LIMIT LEAST(p_match_count * 2, 30)
),
keyword_semantic_search AS ( -- Renamed for clarity
  SELECT
    c.id,
    (v.keyword_vector <#> p_query_keyword_embedding) * -1 as keyword_similarity_score,
    ROW_NUMBER() OVER (ORDER BY (v.keyword_vector <#> p_query_keyword_embedding)) as rank_ix
  FROM public.clips c
  JOIN public.vectors v ON c.id = v.clip_id
  WHERE c.user_id = p_user_id_filter
    AND v.embedding_type = 'full_clip'
    AND v.keyword_vector IS NOT NULL
  ORDER BY rank_ix ASC
  LIMIT LEAST(p_match_count * 2, 30)
)
SELECT
  COALESCE(ft.id, ss.id, ks.id) as id, -- Ensure we get ID from any source
  COALESCE(ft.file_name, ss.file_name, (SELECT cl.file_name FROM public.clips cl WHERE cl.id = ks.id)) as file_name,
  COALESCE(ft.local_path, ss.local_path, (SELECT cl.local_path FROM public.clips cl WHERE cl.id = ks.id)) as local_path,
  COALESCE(ft.content_summary, ss.content_summary, (SELECT cl.content_summary FROM public.clips cl WHERE cl.id = ks.id)) as content_summary,
  COALESCE(ft.content_tags, ss.content_tags, (SELECT cl.content_tags FROM public.clips cl WHERE cl.id = ks.id)) as content_tags,
  COALESCE(ft.duration_seconds, ss.duration_seconds, (SELECT cl.duration_seconds FROM public.clips cl WHERE cl.id = ks.id)) as duration_seconds,
  COALESCE(ft.camera_make, ss.camera_make, (SELECT cl.camera_make FROM public.clips cl WHERE cl.id = ks.id)) as camera_make,
  COALESCE(ft.camera_model, ss.camera_model, (SELECT cl.camera_model FROM public.clips cl WHERE cl.id = ks.id)) as camera_model,
  COALESCE(ft.content_category, ss.content_category, (SELECT cl.content_category FROM public.clips cl WHERE cl.id = ks.id)) as content_category,
  COALESCE(ft.processed_at, ss.processed_at, (SELECT cl.processed_at FROM public.clips cl WHERE cl.id = ks.id)) as processed_at,
  COALESCE(ft.transcript_preview, ss.transcript_preview, (SELECT cl.transcript_preview FROM public.clips cl WHERE cl.id = ks.id)) as transcript_preview,
  COALESCE(ss.similarity_score, ks.keyword_similarity_score, 0.0) as similarity_score, -- Show best semantic score
  -- RRF SCORING WITH DUAL VECTORS AND FULL-TEXT
  (COALESCE(1.0 / (p_rrf_k + ft.rank_ix), 0.0) * p_fulltext_weight) +
  (COALESCE(1.0 / (p_rrf_k + ss.rank_ix), 0.0) * p_summary_weight) +
  (COALESCE(1.0 / (p_rrf_k + ks.rank_ix), 0.0) * p_keyword_weight) as search_rank,
  CASE 
    WHEN ft.id IS NOT NULL AND (ss.id IS NOT NULL OR ks.id IS NOT NULL) THEN 'hybrid'
    WHEN ft.id IS NOT NULL THEN 'fulltext'
    ELSE 'semantic'
  END as match_type
FROM fulltext_search ft
FULL OUTER JOIN summary_semantic_search ss ON ft.id = ss.id
FULL OUTER JOIN keyword_semantic_search ks ON COALESCE(ft.id, ss.id) = ks.id
ORDER BY search_rank DESC
LIMIT p_match_count;
$$;

-- Function for full-text search only
CREATE OR REPLACE FUNCTION public.fulltext_search_clips(
  p_query_text TEXT,
  p_user_id_filter UUID,
  p_match_count INT DEFAULT 10
)
RETURNS TABLE (
  id UUID,
  file_name TEXT,
  local_path TEXT,
  content_summary TEXT,
  content_tags TEXT[],
  duration_seconds NUMERIC,
  camera_make TEXT,
  camera_model TEXT,
  content_category TEXT,
  processed_at TIMESTAMPTZ,
  transcript_preview TEXT,
  fts_rank FLOAT
)
LANGUAGE SQL
SECURITY INVOKER
SET search_path = ''
AS $$
SELECT
  c.id,
  c.file_name,
  c.local_path,
  c.content_summary,
  c.content_tags,
  c.duration_seconds,
  c.camera_make,
  c.camera_model,
  c.content_category,
  c.processed_at,
  c.transcript_preview,
  ts_rank_cd(c.fts, websearch_to_tsquery('english', p_query_text)) as fts_rank
FROM public.clips c
WHERE c.user_id = p_user_id_filter
  AND c.fts @@ websearch_to_tsquery('english', p_query_text)
ORDER BY fts_rank DESC
LIMIT p_match_count;
$$;

-- Function to search transcripts specifically
CREATE OR REPLACE FUNCTION public.search_transcripts(
  p_query_text TEXT,
  p_user_id_filter UUID,
  p_match_count INT DEFAULT 10,
  p_min_content_length INT DEFAULT 50
)
RETURNS TABLE (
  clip_id UUID,
  file_name TEXT,
  local_path TEXT,
  content_summary TEXT,
  full_text TEXT,
  transcript_preview TEXT,
  duration_seconds NUMERIC,
  processed_at TIMESTAMPTZ,
  fts_rank FLOAT
)
LANGUAGE SQL
SECURITY INVOKER
SET search_path = ''
AS $$
SELECT
  t.clip_id,
  c.file_name,
  c.local_path,
  c.content_summary,
  t.full_text,
  c.transcript_preview,
  c.duration_seconds,
  c.processed_at,
  ts_rank_cd(t.fts, websearch_to_tsquery('english', p_query_text)) as fts_rank
FROM public.transcripts t
JOIN public.clips c ON t.clip_id = c.id
WHERE t.user_id = p_user_id_filter
  AND LENGTH(t.full_text) >= p_min_content_length
  AND t.fts @@ websearch_to_tsquery('english', p_query_text)
ORDER BY fts_rank DESC
LIMIT p_match_count;
$$;

-- Function to find similar clips based on existing clip
CREATE OR REPLACE FUNCTION public.find_similar_clips(
  p_source_clip_id UUID,
  p_user_id_filter UUID,
  p_match_count INT DEFAULT 5,
  p_similarity_threshold FLOAT DEFAULT 0.5 -- This threshold is for inner product similarity (higher is better)
)
RETURNS TABLE (
  id UUID,
  file_name TEXT,
  local_path TEXT,
  content_summary TEXT,
  content_tags TEXT[],
  duration_seconds NUMERIC,
  content_category TEXT,
  similarity_score FLOAT
)
LANGUAGE SQL
SECURITY INVOKER
SET search_path = ''
AS $$
WITH source_vector AS (
  SELECT v.summary_vector
  FROM public.vectors v
  WHERE v.clip_id = p_source_clip_id
    AND v.embedding_type = 'full_clip'
    AND v.summary_vector IS NOT NULL
  LIMIT 1
)
SELECT
  c.id,
  c.file_name,
  c.local_path,
  c.content_summary,
  c.content_tags,
  c.duration_seconds,
  c.content_category,
  (v.summary_vector <#> sv.summary_vector) * -1 as similarity_score -- Inner product similarity
FROM public.clips c
JOIN public.vectors v ON c.id = v.clip_id
CROSS JOIN source_vector sv
WHERE c.user_id = p_user_id_filter
  AND c.id != p_source_clip_id
  AND v.embedding_type = 'full_clip'
  AND v.summary_vector IS NOT NULL
  AND (v.summary_vector <#> sv.summary_vector) * -1 >= p_similarity_threshold
ORDER BY similarity_score DESC -- Higher inner product is more similar
LIMIT p_match_count;
$$;