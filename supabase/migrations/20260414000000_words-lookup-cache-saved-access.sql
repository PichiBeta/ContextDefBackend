-- Extend RLS on words_lookup_cache and words_saved to allow access when the user
-- has the reading saved in user_saved_readings.
--
-- Previously these policies only permitted access to "available" readings
-- (is_deleted = false). After soft-delete was introduced, users who have a reading
-- in their library retain access to the reading itself (see read_readings policy in
-- 20260413000001_read-readings-saved-access.sql), but could no longer read, insert,
-- or update cache/vocabulary entries for that reading. This migration adds the same
-- saved-reading OR branch to fix the gap.

--------------------------------------------------------------------------------
-- words_lookup_cache
--------------------------------------------------------------------------------

DROP POLICY IF EXISTS "Users can read lookup cache for readable readings" ON public.words_lookup_cache;
CREATE POLICY "Users can read lookup cache for readable readings"
    ON public.words_lookup_cache FOR SELECT TO authenticated
    USING (EXISTS (
        SELECT 1 FROM public.readings r
        WHERE r.id = words_lookup_cache.reading_id
          AND (
              -- Normal case: reading is live and accessible
              (
                  r.is_deleted = false
                  AND (r.visibility IN ('public', 'unlisted') OR r.owner_id = auth.uid())
              )
              OR
              -- Saved case: caller has this reading in their library (survives soft-delete)
              EXISTS (
                  SELECT 1 FROM public.user_saved_readings usr
                  WHERE usr.reading_id = r.id
                    AND usr.user_id    = auth.uid()
              )
          )
    ));

DROP POLICY IF EXISTS "Users can insert lookup cache for readable readings" ON public.words_lookup_cache;
CREATE POLICY "Users can insert lookup cache for readable readings"
    ON public.words_lookup_cache FOR INSERT TO authenticated
    WITH CHECK (EXISTS (
        SELECT 1 FROM public.readings r
        WHERE r.id = words_lookup_cache.reading_id
          AND (
              (
                  r.is_deleted = false
                  AND (r.visibility IN ('public', 'unlisted') OR r.owner_id = auth.uid())
              )
              OR
              EXISTS (
                  SELECT 1 FROM public.user_saved_readings usr
                  WHERE usr.reading_id = r.id
                    AND usr.user_id    = auth.uid()
              )
          )
    ));

DROP POLICY IF EXISTS "Users can update lookup cache for readable readings" ON public.words_lookup_cache;
CREATE POLICY "Users can update lookup cache for readable readings"
    ON public.words_lookup_cache FOR UPDATE TO authenticated
    USING (EXISTS (
        SELECT 1 FROM public.readings r
        WHERE r.id = words_lookup_cache.reading_id
          AND (
              (
                  r.is_deleted = false
                  AND (r.visibility IN ('public', 'unlisted') OR r.owner_id = auth.uid())
              )
              OR
              EXISTS (
                  SELECT 1 FROM public.user_saved_readings usr
                  WHERE usr.reading_id = r.id
                    AND usr.user_id    = auth.uid()
              )
          )
    ))
    WITH CHECK (EXISTS (
        SELECT 1 FROM public.readings r
        WHERE r.id = words_lookup_cache.reading_id
          AND (
              (
                  r.is_deleted = false
                  AND (r.visibility IN ('public', 'unlisted') OR r.owner_id = auth.uid())
              )
              OR
              EXISTS (
                  SELECT 1 FROM public.user_saved_readings usr
                  WHERE usr.reading_id = r.id
                    AND usr.user_id    = auth.uid()
              )
          )
    ));

--------------------------------------------------------------------------------
-- words_saved
--------------------------------------------------------------------------------

DROP POLICY IF EXISTS "Users can insert their own saved words" ON public.words_saved;
CREATE POLICY "Users can insert their own saved words"
    ON public.words_saved FOR INSERT TO authenticated
    WITH CHECK (
        auth.uid() = user_id
        AND EXISTS (
            SELECT 1 FROM public.readings r
            WHERE r.id = words_saved.reading_id
              AND (
                  (
                      r.is_deleted = false
                      AND (r.visibility IN ('public', 'unlisted') OR r.owner_id = auth.uid())
                  )
                  OR
                  EXISTS (
                      SELECT 1 FROM public.user_saved_readings usr
                      WHERE usr.reading_id = r.id
                        AND usr.user_id    = auth.uid()
                  )
              )
        )
    );

DROP POLICY IF EXISTS "Users can update their own saved words" ON public.words_saved;
CREATE POLICY "Users can update their own saved words"
    ON public.words_saved FOR UPDATE TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (
        auth.uid() = user_id
        AND EXISTS (
            SELECT 1 FROM public.readings r
            WHERE r.id = words_saved.reading_id
              AND (
                  (
                      r.is_deleted = false
                      AND (r.visibility IN ('public', 'unlisted') OR r.owner_id = auth.uid())
                  )
                  OR
                  EXISTS (
                      SELECT 1 FROM public.user_saved_readings usr
                      WHERE usr.reading_id = r.id
                        AND usr.user_id    = auth.uid()
                  )
              )
        )
    );
