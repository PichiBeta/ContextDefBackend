-- Pull two tables created on the dashboard before code-first policy.
-- Uses IF NOT EXISTS / DO $$ guards so this is a no-op on production
-- but creates the tables on local db reset.

--------------------------------------------------------------------------------
-- words_lookup_cache
--------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.words_lookup_cache (
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    selection text NOT NULL,
    definition text NOT NULL,
    translation text NOT NULL,
    reading_id uuid NOT NULL,
    selection_start integer NOT NULL,
    selection_end integer NOT NULL,
    native_language language_code NOT NULL,
    last_accessed timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT words_lookup_cache_pkey PRIMARY KEY (id),
    CONSTRAINT words_lookup_cache_reading_id_fkey
        FOREIGN KEY (reading_id) REFERENCES readings(id) ON DELETE CASCADE,
    CONSTRAINT words_lookup_cache_unique_span
        UNIQUE (reading_id, selection_start, selection_end, native_language),
    CONSTRAINT words_lookup_cache_selection_range_check
        CHECK (selection_start >= 0 AND selection_end >= selection_start)
);

CREATE INDEX IF NOT EXISTS words_lookup_cache_reading_id_native_language_idx
    ON words_lookup_cache USING btree (reading_id, native_language);

ALTER TABLE words_lookup_cache ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'words_lookup_cache'
    AND policyname = 'Users can read lookup cache for readable readings') THEN
CREATE POLICY "Users can read lookup cache for readable readings"
    ON words_lookup_cache FOR SELECT TO authenticated
    USING (EXISTS (
        SELECT 1 FROM readings r
        WHERE r.id = words_lookup_cache.reading_id
          AND r.is_deleted = false
          AND (r.visibility IN ('public', 'unlisted') OR r.owner_id = auth.uid())
    ));
END IF;
END $$;

DO $$ BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'words_lookup_cache'
    AND policyname = 'Users can insert lookup cache for readable readings') THEN
CREATE POLICY "Users can insert lookup cache for readable readings"
    ON words_lookup_cache FOR INSERT TO authenticated
    WITH CHECK (EXISTS (
        SELECT 1 FROM readings r
        WHERE r.id = words_lookup_cache.reading_id
          AND r.is_deleted = false
          AND (r.visibility IN ('public', 'unlisted') OR r.owner_id = auth.uid())
    ));
END IF;
END $$;

DO $$ BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'words_lookup_cache'
    AND policyname = 'Users can update lookup cache for readable readings') THEN
CREATE POLICY "Users can update lookup cache for readable readings"
    ON words_lookup_cache FOR UPDATE TO authenticated
    USING (EXISTS (
        SELECT 1 FROM readings r
        WHERE r.id = words_lookup_cache.reading_id
          AND r.is_deleted = false
          AND (r.visibility IN ('public', 'unlisted') OR r.owner_id = auth.uid())
    ))
    WITH CHECK (EXISTS (
        SELECT 1 FROM readings r
        WHERE r.id = words_lookup_cache.reading_id
          AND r.is_deleted = false
          AND (r.visibility IN ('public', 'unlisted') OR r.owner_id = auth.uid())
    ));
END IF;
END $$;

--------------------------------------------------------------------------------
-- words_saved
--------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.words_saved (
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL,
    reading_id uuid NOT NULL,
    selection text NOT NULL,
    context text NOT NULL,
    definition text NOT NULL,
    translation text NOT NULL,
    native_language language_code NOT NULL,
    selection_start integer NOT NULL,
    selection_end integer NOT NULL,
    saved_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT words_saved_pkey PRIMARY KEY (id),
    CONSTRAINT words_saved_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES profiles(id) ON DELETE CASCADE,
    CONSTRAINT words_saved_reading_id_fkey
        FOREIGN KEY (reading_id) REFERENCES readings(id) ON DELETE CASCADE,
    CONSTRAINT words_saved_unique_user_span
        UNIQUE (user_id, reading_id, selection_start, selection_end),
    CONSTRAINT words_saved_selection_range_check
        CHECK (selection_start >= 0 AND selection_end >= selection_start)
);

ALTER TABLE words_saved ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_words_saved_set_updated_at') THEN
CREATE TRIGGER trg_words_saved_set_updated_at
    BEFORE UPDATE ON words_saved
    FOR EACH ROW EXECUTE FUNCTION handle_updated_at();
END IF;
END $$;

DO $$ BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'words_saved'
    AND policyname = 'Users can view their own saved words') THEN
CREATE POLICY "Users can view their own saved words"
    ON words_saved FOR SELECT TO authenticated
    USING (auth.uid() = user_id);
END IF;
END $$;

DO $$ BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'words_saved'
    AND policyname = 'Users can insert their own saved words') THEN
CREATE POLICY "Users can insert their own saved words"
    ON words_saved FOR INSERT TO authenticated
    WITH CHECK (
        auth.uid() = user_id
        AND EXISTS (
            SELECT 1 FROM readings r
            WHERE r.id = words_saved.reading_id
              AND r.is_deleted = false
              AND (r.visibility IN ('public', 'unlisted') OR r.owner_id = auth.uid())
        )
    );
END IF;
END $$;

DO $$ BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'words_saved'
    AND policyname = 'Users can update their own saved words') THEN
CREATE POLICY "Users can update their own saved words"
    ON words_saved FOR UPDATE TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (
        auth.uid() = user_id
        AND EXISTS (
            SELECT 1 FROM readings r
            WHERE r.id = words_saved.reading_id
              AND r.is_deleted = false
              AND (r.visibility IN ('public', 'unlisted') OR r.owner_id = auth.uid())
        )
    );
END IF;
END $$;

DO $$ BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'words_saved'
    AND policyname = 'Users can delete their own saved words') THEN
CREATE POLICY "Users can delete their own saved words"
    ON words_saved FOR DELETE TO authenticated
    USING (auth.uid() = user_id);
END IF;
END $$;
