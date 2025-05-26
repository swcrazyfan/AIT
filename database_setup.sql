-- =====================================================
-- AI INGESTING TOOL - COMPLETE DATABASE SETUP
-- Version: PRODUCTION READY - All fixes incorporated
-- =====================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- =====================================================
-- 1. USER PROFILES TABLE & TRIGGER (WORKING VERSION)
-- =====================================================
CREATE TABLE IF NOT EXISTS public.user_profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  profile_type TEXT CHECK (profile_type IN ('admin', 'user')) DEFAULT 'user',
  display_name TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;

-- Bulletproof user profile creation trigger (TESTED & WORKING)
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER 
LANGUAGE plpgsql 
SECURITY DEFINER
SET search_path = '' -- Recommended for security
AS $$
BEGIN
  INSERT INTO public.user_profiles (id, profile_type, display_name)
  VALUES (
    NEW.id,
    'user',
    COALESCE(
      NEW.raw_user_meta_data->>'display_name',
      NEW.raw_user_meta_data->>'full_name', 
      NEW.raw_user_meta_data->>'name',
      split_part(NEW.email, '@', 1),
      'User'
    )
  );
  RETURN NEW;
EXCEPTION
  WHEN OTHERS THEN
    RAISE WARNING 'Failed to create user profile for user %: %', NEW.id, SQLERRM;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- =====================================================
-- 2. CLIPS TABLE - Main video storage
-- =====================================================
CREATE TABLE IF NOT EXISTS public.clips (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) NOT NULL,
  
  -- File information
  file_path TEXT NOT NULL,
  local_path TEXT NOT NULL,
  file_name TEXT NOT NULL,
  file_checksum TEXT UNIQUE NOT NULL,
  file_size_bytes BIGINT NOT NULL,
  duration_seconds NUMERIC,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(), -- Added updated_at column
  processed_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Technical metadata
  width INTEGER,
  height INTEGER,
  frame_rate NUMERIC,
  codec TEXT,
  camera_make TEXT,
  camera_model TEXT,
  container TEXT,
  
  -- Processing metadata
  processing_status JSONB DEFAULT '{}',
  processing_progress INTEGER DEFAULT 0,
  total_steps INTEGER DEFAULT 0,
  current_step TEXT,
  last_step_completed TEXT,
  processed_steps TEXT[] DEFAULT '{}', -- Added processed_steps column
  
  -- AI analysis summaries
  content_category TEXT,
  content_summary TEXT,
  content_tags TEXT[],
  
  -- Transcript data
  full_transcript TEXT,
  transcript_preview TEXT,
  transcript_word_count INTEGER DEFAULT 0,
  scene_count INTEGER DEFAULT 0,
  ai_processing_status TEXT,
  
  -- Search columns (populated by triggers)
  searchable_content TEXT,
  fts tsvector,
  
  -- Complex metadata as JSONB
  technical_metadata JSONB,
  camera_details JSONB,
  audio_tracks JSONB,
  subtitle_tracks JSONB,
  thumbnails TEXT[]
);

ALTER TABLE public.clips ENABLE ROW LEVEL SECURITY;

-- Search content trigger for clips
CREATE OR REPLACE FUNCTION public.update_clips_search_content()
RETURNS TRIGGER 
LANGUAGE plpgsql
SET search_path = '' -- Recommended for security
AS $$
BEGIN
  NEW.searchable_content := COALESCE(NEW.file_name, '') || ' ' ||
                           COALESCE(NEW.content_summary, '') || ' ' ||
                           COALESCE(NEW.transcript_preview, '') || ' ' ||
                           COALESCE(array_to_string(NEW.content_tags, ' '), '') || ' ' ||
                           COALESCE(NEW.content_category, '');
  NEW.fts := to_tsvector('english', NEW.searchable_content);
  RETURN NEW;
END;
$$;

CREATE TRIGGER clips_search_content_trigger
  BEFORE INSERT OR UPDATE ON public.clips
  FOR EACH ROW EXECUTE FUNCTION public.update_clips_search_content();

-- Add processing columns if they don't exist (idempotent)
ALTER TABLE public.clips ADD COLUMN IF NOT EXISTS processing_status JSONB DEFAULT '{}';
ALTER TABLE public.clips ADD COLUMN IF NOT EXISTS processing_progress INTEGER DEFAULT 0;
ALTER TABLE public.clips ADD COLUMN IF NOT EXISTS total_steps INTEGER DEFAULT 0;
ALTER TABLE public.clips ADD COLUMN IF NOT EXISTS current_step TEXT;
ALTER TABLE public.clips ADD COLUMN IF NOT EXISTS last_step_completed TEXT;
ALTER TABLE public.clips ADD COLUMN IF NOT EXISTS ai_processing_status TEXT;
ALTER TABLE public.clips ADD COLUMN IF NOT EXISTS scene_count INTEGER DEFAULT 0;
ALTER TABLE public.clips ADD COLUMN IF NOT EXISTS transcript_preview TEXT;
ALTER TABLE public.clips ADD COLUMN IF NOT EXISTS transcript_word_count INTEGER DEFAULT 0;
ALTER TABLE public.clips ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW(); -- Ensure column exists if table already created

-- Trigger function to automatically update 'updated_at' on clips table
CREATE OR REPLACE FUNCTION public.handle_updated_at_clips()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER -- Can be invoker as it only modifies the row being updated
SET search_path = ''
AS $$
BEGIN
  NEW.updated_at := NOW();
  RETURN NEW;
END;
$$;

-- Trigger to call the function before any update on the clips table
DROP TRIGGER IF EXISTS trigger_clips_updated_at ON public.clips; -- Drop if exists to avoid errors on re-run
CREATE TRIGGER trigger_clips_updated_at
  BEFORE UPDATE ON public.clips
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_updated_at_clips();

-- =====================================================
-- 3. PROCESSING EVENTS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS public.processing_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES public.clips(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    step_name TEXT NOT NULL,
    step_index INTEGER,
    status TEXT NOT NULL,
    error TEXT,
    timestamp TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}'
);

ALTER TABLE public.processing_events ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 4. SCENES TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS public.scenes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES public.clips(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    scene_number INTEGER NOT NULL,
    timestamp_start FLOAT,
    timestamp_end FLOAT,
    description TEXT,
    objects TEXT[],
    activities TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(video_id, scene_number)
);

ALTER TABLE public.scenes ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 5. ARTIFACTS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS public.artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES public.clips(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    step_name TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

ALTER TABLE public.artifacts ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 6. VECTORS TABLE - For semantic search
-- =====================================================
CREATE TABLE IF NOT EXISTS public.vectors (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  clip_id UUID REFERENCES public.clips(id) ON DELETE CASCADE,
  -- segment_id UUID REFERENCES public.segments(id) ON DELETE CASCADE, -- Segments table not defined in this plan section
  segment_id UUID, -- Placeholder: Uncomment above if segments table is added
  user_id UUID REFERENCES auth.users(id) NOT NULL,
  embedding_type TEXT NOT NULL CHECK (embedding_type IN ('full_clip', 'segment', 'keyframe')),
  embedding_source TEXT NOT NULL,
  summary_vector vector(1024),
  keyword_vector vector(1024),
  embedded_content TEXT NOT NULL,
  original_content TEXT,
  token_count INTEGER,
  original_token_count INTEGER,
  truncation_method TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  CONSTRAINT check_vector_scope CHECK (
    (embedding_type = 'full_clip' AND segment_id IS NULL) OR
    (embedding_type IN ('segment', 'keyframe') AND segment_id IS NOT NULL)
  )
);

ALTER TABLE public.vectors ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 7. TRANSCRIPTS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS public.transcripts (
  clip_id UUID REFERENCES public.clips(id) ON DELETE CASCADE,
  user_id UUID REFERENCES auth.users(id) NOT NULL,
  full_text TEXT NOT NULL,
  segments JSONB NOT NULL,
  speakers JSONB,
  non_speech_events JSONB,
  fts tsvector,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (clip_id)
);

ALTER TABLE public.transcripts ENABLE ROW LEVEL SECURITY;

-- Transcript search trigger
CREATE OR REPLACE FUNCTION public.update_transcript_search()
RETURNS TRIGGER 
LANGUAGE plpgsql
SET search_path = '' -- Recommended for security
AS $$
BEGIN
  NEW.fts := to_tsvector('english', NEW.full_text);
  RETURN NEW;
END;
$$;

CREATE TRIGGER transcripts_search_trigger
  BEFORE INSERT OR UPDATE ON public.transcripts
  FOR EACH ROW EXECUTE FUNCTION public.update_transcript_search();

-- =====================================================
-- 8. ANALYSIS TABLE - AI analysis results
-- =====================================================
CREATE TABLE IF NOT EXISTS public.analysis (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  clip_id UUID REFERENCES public.clips(id) ON DELETE CASCADE,
  -- segment_id UUID REFERENCES public.segments(id) ON DELETE CASCADE, -- Segments table not defined in this plan section
  segment_id UUID, -- Placeholder: Uncomment above if segments table is added
  user_id UUID REFERENCES auth.users(id) NOT NULL,
  analysis_type TEXT NOT NULL,
  analysis_scope TEXT NOT NULL CHECK (analysis_scope IN ('full_clip', 'segment')),
  ai_model TEXT DEFAULT 'gemini-flash-2.5',
  content_category TEXT,
  usability_rating TEXT,
  speaker_count INTEGER,
  visual_analysis JSONB,
  audio_analysis JSONB,
  content_analysis JSONB,
  analysis_summary JSONB,
  analysis_file_path TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  CONSTRAINT check_analysis_scope CHECK (
    (analysis_scope = 'full_clip' AND segment_id IS NULL) OR
    (analysis_scope = 'segment' AND segment_id IS NOT NULL)
  )
);

ALTER TABLE public.analysis ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 9. ROW LEVEL SECURITY POLICIES
-- =====================================================

-- User Profile Policies
CREATE POLICY "Users can view own profile" ON public.user_profiles
    FOR SELECT USING (id = auth.uid());

CREATE POLICY "Users can update own profile" ON public.user_profiles
    FOR UPDATE USING (id = auth.uid())
    WITH CHECK (id = auth.uid());

CREATE POLICY "Admins can view all profiles" ON public.user_profiles
    FOR SELECT USING (public.is_admin()); -- Use the is_admin() helper function

-- Clips Policies
CREATE POLICY "Users can view own clips" ON public.clips 
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert own clips" ON public.clips 
    FOR INSERT WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can update own clips" ON public.clips 
    FOR UPDATE USING (user_id = auth.uid()) 
    WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can delete own clips" ON public.clips 
    FOR DELETE USING (user_id = auth.uid());

CREATE POLICY "Admins can view all clips" ON public.clips 
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.user_profiles 
            WHERE id = auth.uid() AND profile_type = 'admin'
        )
    );

-- Processing Events Policies
CREATE POLICY "Users can view own processing events" ON public.processing_events 
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert own processing events" ON public.processing_events 
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- Scenes Policies
CREATE POLICY "Users can view own scenes" ON public.scenes 
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert own scenes" ON public.scenes 
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- Artifacts Policies
CREATE POLICY "Users can view own artifacts" ON public.artifacts 
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert own artifacts" ON public.artifacts 
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- Vectors Policies
CREATE POLICY "Users can view own vectors" ON public.vectors 
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert own vectors" ON public.vectors 
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- Transcripts Policies
CREATE POLICY "Users can view own transcripts" ON public.transcripts 
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert own transcripts" ON public.transcripts 
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- Analysis Policies
CREATE POLICY "Users can view own analysis" ON public.analysis 
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert own analysis" ON public.analysis 
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- =====================================================
-- 10. PERFORMANCE INDEXES
-- =====================================================
CREATE INDEX IF NOT EXISTS idx_clips_user_id ON public.clips(user_id);
CREATE INDEX IF NOT EXISTS idx_clips_processing_status ON public.clips(processing_progress, current_step);
CREATE INDEX IF NOT EXISTS idx_processing_events_video ON public.processing_events(video_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_processing_events_user ON public.processing_events(user_id);
CREATE INDEX IF NOT EXISTS idx_scenes_video ON public.scenes(video_id, scene_number);
CREATE INDEX IF NOT EXISTS idx_artifacts_video ON public.artifacts(video_id, step_name);

CREATE INDEX IF NOT EXISTS idx_clips_fts ON public.clips USING gin(fts);
CREATE INDEX IF NOT EXISTS idx_transcripts_fts ON public.transcripts USING gin(fts);
CREATE INDEX IF NOT EXISTS idx_vectors_summary ON public.vectors USING hnsw (summary_vector vector_ip_ops);
CREATE INDEX IF NOT EXISTS idx_vectors_keyword ON public.vectors USING hnsw (keyword_vector vector_ip_ops);

-- =====================================================
-- 11. HELPER FUNCTIONS
-- =====================================================

-- Check if user is admin
CREATE OR REPLACE FUNCTION public.is_admin(check_user_id UUID DEFAULT auth.uid())
RETURNS BOOLEAN
LANGUAGE SQL
SECURITY DEFINER -- Requires DEFINER to check other users' profiles
AS $$
    SELECT EXISTS (
        SELECT 1 FROM public.user_profiles 
        WHERE id = check_user_id AND profile_type = 'admin'
    );
$$;

-- Get user profile info
CREATE OR REPLACE FUNCTION public.get_user_profile(profile_user_id UUID DEFAULT auth.uid())
RETURNS TABLE (
    id UUID,
    profile_type TEXT,
    display_name TEXT,
    created_at TIMESTAMPTZ
)
LANGUAGE SQL
SECURITY INVOKER -- Changed to INVOKER to respect RLS policies
AS $$
    SELECT up.id, up.profile_type, up.display_name, up.created_at
    FROM public.user_profiles up
    WHERE up.id = profile_user_id;
$$;

-- Get user statistics
CREATE OR REPLACE FUNCTION public.get_user_stats(stats_user_id UUID DEFAULT auth.uid())
RETURNS TABLE (
    total_clips INTEGER,
    total_duration_hours NUMERIC,
    total_storage_gb NUMERIC,
    clips_with_transcripts INTEGER,
    clips_in_progress INTEGER,
    clips_completed INTEGER
)
LANGUAGE SQL
SECURITY DEFINER -- Requires DEFINER to aggregate stats across a user's data,
                 -- potentially bypassing RLS on clips/transcripts for counting.
AS $$
    SELECT 
        COUNT(*)::INTEGER as total_clips,
        ROUND(SUM(COALESCE(c.duration_seconds, 0)) / 3600.0, 2) as total_duration_hours,
        ROUND(SUM(COALESCE(c.file_size_bytes, 0)) / (1024.0^3), 2) as total_storage_gb,
        COUNT(CASE WHEN c.transcript_word_count > 0 THEN 1 END)::INTEGER as clips_with_transcripts,
        COUNT(CASE WHEN c.processing_progress < 100 THEN 1 END)::INTEGER as clips_in_progress,
        COUNT(CASE WHEN c.processing_progress = 100 THEN 1 END)::INTEGER as clips_completed
    FROM public.clips c
    WHERE c.user_id = stats_user_id;
$$;

-- Function to append to array columns
CREATE OR REPLACE FUNCTION public.append_to_array(
    p_table_name TEXT,
    p_id UUID,
    p_column_name TEXT,
    p_new_value TEXT
) 
RETURNS VOID 
LANGUAGE plpgsql
SECURITY INVOKER -- Typically invoker is fine for utility functions modifying specific rows owned by user
SET search_path = '' -- Recommended for security
AS $$
BEGIN
    EXECUTE format(
        'UPDATE public.%I SET %I = array_append(COALESCE(%I, ''{}''), %L) WHERE id = %L AND user_id = %L::uuid',
        p_table_name, p_column_name, p_column_name, p_new_value, p_id, auth.uid() -- Added user_id check
    );
END;
$$;

-- View for monitoring active processing
CREATE OR REPLACE VIEW public.active_processing AS
SELECT 
    c.id,
    c.file_name,
    c.user_id,
    c.processing_progress,
    c.current_step,
    c.created_at,
    c.updated_at, 
    EXTRACT(EPOCH FROM (NOW() - c.updated_at)) as seconds_since_update,
    c.processing_status,
    CASE 
        WHEN c.processing_progress = 100 THEN 'completed'
        WHEN EXTRACT(EPOCH FROM (NOW() - c.updated_at)) > 300 THEN 'stalled' -- 5 minutes
        WHEN (c.processing_status->>'error' IS NOT NULL) OR (c.processing_status->(c.current_step)->>'status' = 'failed') OR (c.processing_status->(c.current_step)->>'status' = 'error') THEN 'error'
        ELSE 'processing'
    END as status
FROM public.clips c
WHERE c.processing_progress < 100 
   OR c.updated_at > NOW() - INTERVAL '1 hour' -- Show recently completed/failed as well
ORDER BY c.created_at DESC;

COMMENT ON TABLE public.user_profiles IS 'Stores user profile information, extending auth.users.';
COMMENT ON TABLE public.clips IS 'Main table for storing video metadata and processing status.';
COMMENT ON TABLE public.processing_events IS 'Logs events related to each step of the video processing pipeline.';
COMMENT ON TABLE public.scenes IS 'Stores information about detected scenes within videos.';
COMMENT ON TABLE public.artifacts IS 'Stores paths to file artifacts generated during processing steps (e.g., thumbnails, compressed videos).';
COMMENT ON TABLE public.vectors IS 'Stores vector embeddings for videos and segments for semantic search.';
COMMENT ON TABLE public.transcripts IS 'Stores full transcripts and segmented transcript data for videos.';
COMMENT ON TABLE public.analysis IS 'Stores detailed AI analysis results for videos or video segments.';