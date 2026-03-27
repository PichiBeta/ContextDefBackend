-- Performance and security fixes identified during schema review.

--------------------------------------------------------------------------------
-- 1. HNSW vector indexes for embedding similarity search
--------------------------------------------------------------------------------

CREATE INDEX idx_profiles_embedding ON profiles
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_readings_embedding ON readings
    USING hnsw (embedding vector_cosine_ops)
    WHERE status = 'processed' AND is_deleted = false;

--------------------------------------------------------------------------------
-- 2. Tighten profiles SELECT RLS to authenticated only
--------------------------------------------------------------------------------

DROP POLICY IF EXISTS "Public profiles are viewable by everyone." ON profiles;
CREATE POLICY "Authenticated users can view profiles"
    ON profiles FOR SELECT TO authenticated USING (true);

--------------------------------------------------------------------------------
-- 3. readings.owner_id: allow NULL + ON DELETE SET NULL
--    Preserves readings (especially public ones) when a profile is deleted.
--------------------------------------------------------------------------------

ALTER TABLE readings ALTER COLUMN owner_id DROP NOT NULL;
ALTER TABLE readings DROP CONSTRAINT readings_owner_id_fkey;
ALTER TABLE readings
    ADD CONSTRAINT readings_owner_id_fkey
    FOREIGN KEY (owner_id) REFERENCES profiles(id) ON DELETE SET NULL;
