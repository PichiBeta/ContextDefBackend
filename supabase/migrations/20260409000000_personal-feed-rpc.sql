-- Add get_personal_feed RPC function.
-- Returns a personalised feed for users with an embedding, or a recency feed for new users.

CREATE OR REPLACE FUNCTION public.get_personal_feed(
    p_user_embedding vector,
    p_limit integer DEFAULT 20,
    p_offset integer DEFAULT 0
)
RETURNS TABLE(
    id uuid,
    title text,
    genre text,
    language_code language_code,
    content_preview character varying,
    difficulty integer,
    owner_id uuid,
    created_at timestamp with time zone,
    similarity double precision
)
LANGUAGE plpgsql
STABLE
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
      ORDER BY r.embedding <=> p_user_embedding
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
