-- Creates the "avatars" storage bucket and scoped RLS policies.
-- Avatar files are stored at {userId}/avatar.{ext} (e.g. abc-123/avatar.jpg).
-- Public bucket: any authenticated user may read; users may only write their own avatar.

-- 1. Bucket
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types, avif_autodetection)
VALUES (
  'avatars',
  'avatars',
  true,
  10485760,           -- 10 MB
  ARRAY['image/*'],
  false
)
ON CONFLICT (id) DO NOTHING;

-- 2. RLS (no-op if already enabled; storage.objects has RLS on by default in Supabase)
ALTER TABLE storage.objects ENABLE ROW LEVEL SECURITY;

-- 3. Policies

-- Any authenticated user can read any avatar
CREATE OR REPLACE POLICY "avatars_select"
  ON storage.objects
  AS PERMISSIVE
  FOR SELECT
  TO authenticated
  USING (bucket_id = 'avatars');

-- Authenticated users may only INSERT into their own folder
CREATE OR REPLACE POLICY "avatars_insert"
  ON storage.objects
  AS PERMISSIVE
  FOR INSERT
  TO authenticated
  WITH CHECK (
    bucket_id = 'avatars'
    AND (storage.foldername(name))[1] = auth.uid()::text
  );

-- Authenticated users may only UPDATE their own avatar.
-- WITH CHECK mirrors USING to prevent renaming the file into another user's folder.
CREATE OR REPLACE POLICY "avatars_update"
  ON storage.objects
  AS PERMISSIVE
  FOR UPDATE
  TO authenticated
  USING (
    bucket_id = 'avatars'
    AND (storage.foldername(name))[1] = auth.uid()::text
  )
  WITH CHECK (
    bucket_id = 'avatars'
    AND (storage.foldername(name))[1] = auth.uid()::text
  );

-- Authenticated users may only DELETE their own avatar
CREATE OR REPLACE POLICY "avatars_delete"
  ON storage.objects
  AS PERMISSIVE
  FOR DELETE
  TO authenticated
  USING (
    bucket_id = 'avatars'
    AND (storage.foldername(name))[1] = auth.uid()::text
  );
