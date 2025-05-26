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
CREATE TABLE IF NOT EXISTS user_profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  profile_type TEXT CHECK (profile_type IN ('admin', 'user')) DEFAULT 'user',
  display_name TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

-- Bulletproof user profile creation trigger (TESTED & WORKING)
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER 
LANGUAGE plpgsql 
SECURITY DEFINER
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
  FOR EACH ROW EXECUTE FUNCTION handle_new_user();

-- =====================================================
-- 2. CLIPS TABLE - Main video storage
-- =====================================================
CREATE TABLE IF NOT EXISTS clips (
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
  processed_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Technical metadata
  width INTEGER,
  height INTEGER,
  frame_rate NUMERIC,
  codec TEXT,
  camera_make TEXT,
  camera_model TEXT,
  container TEXT,
  
  -- AI analysis summaries
  content_category TEXT,
  content_summary TEXT,
  content_tags TEXT[],
  
  -- Transcript data
  full_transcript TEXT,
  transcript_preview TEXT,
  
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

ALTER TABLE clips ENABLE ROW LEVEL SECURITY;
-- Search content trigger for clips
CREATE OR REPLACE FUNCTION update_clips_search_content()
RETURNS TRIGGER AS $$
BEGIN
  NEW.searchable_content := COALESCE(NEW.file_name, '') || ' ' ||
                           COALESCE(NEW.content_summary, '') || ' ' ||
                           COALESCE(NEW.transcript_preview, '') || ' ' ||
                           COALESCE(array_to_string(NEW.content_tags, ' '), '') || ' ' ||
                           COALESCE(NEW.content_category, '');
  NEW.fts := to_tsvector('english', NEW.searchable_content);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER clips_search_content_trigger
  BEFORE INSERT OR UPDATE ON clips
  FOR EACH ROW EXECUTE FUNCTION update_clips_search_content();

-- =====================================================
-- 3. SEGMENTS TABLE - For future segment-level analysis
-- =====================================================
CREATE TABLE IF NOT EXISTS segments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  clip_id UUID REFERENCES clips(id) ON DELETE CASCADE,
  user_id UUID REFERENCES auth.users(id) NOT NULL,
  
  segment_index INTEGER NOT NULL,
  start_time_seconds NUMERIC NOT NULL,
  end_time_seconds NUMERIC NOT NULL,
  duration_seconds NUMERIC,
  
  segment_type TEXT DEFAULT 'auto',
  speaker_id TEXT,
  segment_description TEXT,
  keyframe_timestamp NUMERIC,
  
  segment_content TEXT,
  fts tsvector,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(clip_id, segment_index),
  CONSTRAINT check_segment_times CHECK (start_time_seconds < end_time_seconds),
  CONSTRAINT check_segment_index CHECK (segment_index >= 0)
);

ALTER TABLE segments ENABLE ROW LEVEL SECURITY;

-- Segment calculated fields trigger
CREATE OR REPLACE FUNCTION update_segments_calculated_fields()
RETURNS TRIGGER AS $$
BEGIN
  NEW.duration_seconds := NEW.end_time_seconds - NEW.start_time_seconds;
  NEW.fts := to_tsvector('english', COALESCE(NEW.segment_content, ''));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER segments_calculated_fields_trigger
  BEFORE INSERT OR UPDATE ON segments
  FOR EACH ROW EXECUTE FUNCTION update_segments_calculated_fields();
-- =====================================================
-- 4. ANALYSIS TABLE - AI analysis results
-- =====================================================
CREATE TABLE IF NOT EXISTS analysis (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  clip_id UUID REFERENCES clips(id) ON DELETE CASCADE,
  segment_id UUID REFERENCES segments(id) ON DELETE CASCADE,
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

ALTER TABLE analysis ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 5. VECTORS TABLE - For semantic search
-- =====================================================
CREATE TABLE IF NOT EXISTS vectors (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  clip_id UUID REFERENCES clips(id) ON DELETE CASCADE,
  segment_id UUID REFERENCES segments(id) ON DELETE CASCADE,
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

ALTER TABLE vectors ENABLE ROW LEVEL SECURITY;
-- =====================================================
-- 6. TRANSCRIPTS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS transcripts (
  clip_id UUID REFERENCES clips(id) ON DELETE CASCADE,
  user_id UUID REFERENCES auth.users(id) NOT NULL,
  
  full_text TEXT NOT NULL,
  segments JSONB NOT NULL,
  speakers JSONB,
  non_speech_events JSONB,
  
  fts tsvector,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (clip_id)
);

ALTER TABLE transcripts ENABLE ROW LEVEL SECURITY;

-- Transcript search trigger
CREATE OR REPLACE FUNCTION update_transcript_search()
RETURNS TRIGGER AS $$
BEGIN
  NEW.fts := to_tsvector('english', NEW.full_text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER transcripts_search_trigger
  BEFORE INSERT OR UPDATE ON transcripts
  FOR EACH ROW EXECUTE FUNCTION update_transcript_search();

-- =====================================================
-- 7. PERFORMANCE INDEXES
-- =====================================================
CREATE INDEX IF NOT EXISTS idx_clips_fts ON clips USING gin(fts);
CREATE INDEX IF NOT EXISTS idx_segments_fts ON segments USING gin(fts);
CREATE INDEX IF NOT EXISTS idx_transcripts_fts ON transcripts USING gin(fts);
CREATE INDEX IF NOT EXISTS idx_vectors_summary ON vectors USING hnsw (summary_vector vector_ip_ops);
CREATE INDEX IF NOT EXISTS idx_vectors_keyword ON vectors USING hnsw (keyword_vector vector_ip_ops);
CREATE INDEX IF NOT EXISTS idx_segments_clip_order ON segments(clip_id, segment_index);
CREATE INDEX IF NOT EXISTS idx_segments_time_range ON segments(clip_id, start_time_seconds, end_time_seconds);
CREATE INDEX IF NOT EXISTS idx_clips_user_category ON clips(user_id, content_category);
CREATE INDEX IF NOT EXISTS idx_clips_tags ON clips USING gin(content_tags);
CREATE INDEX IF NOT EXISTS idx_clips_camera ON clips(camera_make, camera_model) WHERE camera_make IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_clips_duration ON clips(duration_seconds) WHERE duration_seconds IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_analysis_clip_type ON analysis(clip_id, analysis_type);
CREATE INDEX IF NOT EXISTS idx_analysis_segment ON analysis(segment_id) WHERE segment_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_vectors_clip_source ON vectors(clip_id, embedding_source);
CREATE INDEX IF NOT EXISTS idx_vectors_segment ON vectors(segment_id) WHERE segment_id IS NOT NULL;
-- =====================================================
-- 8. ROW LEVEL SECURITY POLICIES
-- =====================================================

-- User Profile Policies
DROP POLICY IF EXISTS "Users can view own profile" ON user_profiles;
CREATE POLICY "Users can view own profile" ON user_profiles
  FOR SELECT TO authenticated USING (id = auth.uid());

DROP POLICY IF EXISTS "Users can update own profile" ON user_profiles;
CREATE POLICY "Users can update own profile" ON user_profiles
  FOR UPDATE TO authenticated 
  USING (id = auth.uid())
  WITH CHECK (id = auth.uid() AND profile_type = (SELECT profile_type FROM user_profiles WHERE id = auth.uid()));

DROP POLICY IF EXISTS "Admins can view all profiles" ON user_profiles;
CREATE POLICY "Admins can view all profiles" ON user_profiles
  FOR SELECT TO authenticated
  USING (EXISTS (SELECT 1 FROM user_profiles WHERE id = auth.uid() AND profile_type = 'admin'));

-- Clips Policies
CREATE POLICY "Users can view own clips" ON clips FOR SELECT TO authenticated USING (user_id = auth.uid());
CREATE POLICY "Users can insert own clips" ON clips FOR INSERT TO authenticated WITH CHECK (user_id = auth.uid());
CREATE POLICY "Users can update own clips" ON clips FOR UPDATE TO authenticated USING (user_id = auth.uid()) WITH CHECK (user_id = auth.uid());
CREATE POLICY "Users can delete own clips" ON clips FOR DELETE TO authenticated USING (user_id = auth.uid());
CREATE POLICY "Admins can view all clips" ON clips FOR SELECT TO authenticated USING (EXISTS (SELECT 1 FROM user_profiles WHERE id = auth.uid() AND profile_type = 'admin'));

-- Segments Policies
CREATE POLICY "Users can view own segments" ON segments FOR SELECT TO authenticated USING (user_id = auth.uid());
CREATE POLICY "Users can insert own segments" ON segments FOR INSERT TO authenticated WITH CHECK (user_id = auth.uid() AND EXISTS (SELECT 1 FROM clips WHERE id = clip_id AND user_id = auth.uid()));
CREATE POLICY "Users can update own segments" ON segments FOR UPDATE TO authenticated USING (user_id = auth.uid()) WITH CHECK (user_id = auth.uid());
CREATE POLICY "Users can delete own segments" ON segments FOR DELETE TO authenticated USING (user_id = auth.uid());

-- Analysis Policies
CREATE POLICY "Users can view own analysis" ON analysis FOR SELECT TO authenticated USING (user_id = auth.uid());
CREATE POLICY "Users can insert own analysis" ON analysis FOR INSERT TO authenticated WITH CHECK (user_id = auth.uid() AND EXISTS (SELECT 1 FROM clips WHERE id = clip_id AND user_id = auth.uid()));

-- Vectors Policies
CREATE POLICY "Users can view own vectors" ON vectors FOR SELECT TO authenticated USING (user_id = auth.uid());
CREATE POLICY "Users can insert own vectors" ON vectors FOR INSERT TO authenticated WITH CHECK (user_id = auth.uid() AND EXISTS (SELECT 1 FROM clips WHERE id = clip_id AND user_id = auth.uid()));

-- Transcripts Policies
CREATE POLICY "Users can view own transcripts" ON transcripts FOR SELECT TO authenticated USING (user_id = auth.uid());
CREATE POLICY "Users can insert own transcripts" ON transcripts FOR INSERT TO authenticated WITH CHECK (user_id = auth.uid() AND EXISTS (SELECT 1 FROM clips WHERE id = clip_id AND user_id = auth.uid()));
-- =====================================================
-- 9. HELPER FUNCTIONS (FIXED - no parameter conflicts)
-- =====================================================

-- Drop existing functions to avoid conflicts
DROP FUNCTION IF EXISTS is_admin(UUID);
DROP FUNCTION IF EXISTS get_user_profile(UUID);
DROP FUNCTION IF EXISTS get_user_stats(UUID);

-- Check if user is admin
CREATE FUNCTION is_admin(check_user_id UUID DEFAULT auth.uid())
RETURNS BOOLEAN
LANGUAGE SQL
SECURITY DEFINER
AS $$
  SELECT EXISTS (
    SELECT 1 FROM user_profiles 
    WHERE id = check_user_id AND profile_type = 'admin'
  );
$$;

-- Get user profile info
CREATE FUNCTION get_user_profile(profile_user_id UUID DEFAULT auth.uid())
RETURNS TABLE (
  id UUID,
  profile_type TEXT,
  display_name TEXT,
  created_at TIMESTAMPTZ
)
LANGUAGE SQL
SECURITY DEFINER
AS $$
  SELECT up.id, up.profile_type, up.display_name, up.created_at
  FROM user_profiles up
  WHERE up.id = profile_user_id;
$$;

-- User stats function (FIXED - no ambiguous column names)
CREATE FUNCTION get_user_stats(stats_user_id UUID DEFAULT auth.uid())
RETURNS TABLE (
  total_clips INTEGER,
  total_duration_hours NUMERIC,
  total_storage_gb NUMERIC,
  clips_with_transcripts INTEGER,
  clips_with_ai_analysis INTEGER
)
LANGUAGE SQL
SECURITY DEFINER
AS $$
  SELECT 
    COUNT(*)::INTEGER as total_clips,
    ROUND(SUM(COALESCE(duration_seconds, 0)) / 3600.0, 2) as total_duration_hours,
    ROUND(SUM(COALESCE(file_size_bytes, 0)) / (1024.0^3), 2) as total_storage_gb,
    COUNT(t.clip_id)::INTEGER as clips_with_transcripts,
    COUNT(DISTINCT a.clip_id)::INTEGER as clips_with_ai_analysis
  FROM clips c
  LEFT JOIN transcripts t ON c.id = t.clip_id
  LEFT JOIN analysis a ON c.id = a.clip_id
  WHERE c.user_id = stats_user_id;
$$;

-- =====================================================
-- 10. SETUP COMPLETE MESSAGE
-- =====================================================
DO $$
BEGIN
  RAISE NOTICE '🎉 AI Ingesting Tool Database Setup Complete!';
  RAISE NOTICE '📊 Tables: user_profiles, clips, segments, analysis, vectors, transcripts';
  RAISE NOTICE '🔧 Triggers: Auto user profiles, search content updates';
  RAISE NOTICE '🔒 Security: RLS policies with user isolation + admin override';
  RAISE NOTICE '🚀 Ready for AI Ingesting Tool integration!';
END;
$$;