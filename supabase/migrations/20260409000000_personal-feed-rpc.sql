-- Adds the get_personal_feed RPC used by the personal-feed edge function.
--
-- Two modes, selected by whether p_user_embedding is NULL:
--   Personalized — ORDER BY cosine distance (uses idx_readings_embedding HNSW index)
--   Discovery    — ORDER BY created_at DESC  (uses idx_readings_feed_created_at)
--
-- Saved readings are excluded via NOT EXISTS so a user never sees content they
-- already have in their library.
--
-- SECURITY INVOKER ensures auth.uid() resolves to the calling user's JWT.
-- GRANT EXECUTE is given to the `authenticated` role only.

CREATE OR REPLACE FUNCTION public.get_personal_feed(
  p_user_embedding vector(384),   -- NULL when user has no embedding yet
  p_limit          int DEFAULT 20,
  p_offset         int DEFAULT 0
)
RETURNS TABLE (
  id               uuid,
  title            text,
  genre            text,
  language_code    language_code,
  content_preview  varchar(70),
  difficulty       int,
  owner_id         uuid,
  created_at       timestamptz,
  similarity       float8          -- NULL in discovery mode
)
LANGUAGE plpgsql
STABLE
SECURITY INVOKER
AS $$
BEGIN
  IF p_user_embedding IS NOT NULL THEN
    -- Personalized path: planner sees a plain ORDER BY distance expression so the
    -- HNSW index (idx_readings_embedding) can be used for the top-k scan.
    RETURN QUERY
      SELECT
        r.id,
        r.title,
        r.genre,
        r.language_code,
        r.content_preview,
        r.difficulty,
        r.owner_id,
        r.created_at,
        (1.0 - (r.embedding <=> p_user_embedding))::float8 AS similarity
      FROM readings r
      WHERE
        r.visibility  = 'public'
        AND r.status   = 'processed'
        AND r.is_deleted = false
        AND r.embedding IS NOT NULL
        AND NOT EXISTS (
          SELECT 1
          FROM user_saved_readings usr
          WHERE usr.user_id   = auth.uid()
            AND usr.reading_id = r.id
        )
      ORDER BY r.embedding <=> p_user_embedding  -- ascending = most similar first
      LIMIT  p_limit
      OFFSET p_offset;
  ELSE
    -- Discovery path: new users with no saved readings get a recency feed.
    -- Uses idx_readings_feed_created_at (partial btree on public+processed+not-deleted).
    RETURN QUERY
      SELECT
        r.id,
        r.title,
        r.genre,
        r.language_code,
        r.content_preview,
        r.difficulty,
        r.owner_id,
        r.created_at,
        NULL::float8 AS similarity
      FROM readings r
      WHERE
        r.visibility  = 'public'
        AND r.status   = 'processed'
        AND r.is_deleted = false
        AND NOT EXISTS (
          SELECT 1
          FROM user_saved_readings usr
          WHERE usr.user_id   = auth.uid()
            AND usr.reading_id = r.id
        )
      ORDER BY r.created_at DESC
      LIMIT  p_limit
      OFFSET p_offset;
  END IF;
END;
$$;

GRANT EXECUTE ON FUNCTION public.get_personal_feed(vector, int, int) TO authenticated;
