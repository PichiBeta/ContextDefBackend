-- Allow users who have saved a reading to still read it after the owner soft-deletes it.
-- Also updates the storage read policy for the same reason.
--
-- Previous read_readings policy only allowed access when is_deleted = false.
-- A soft-deleted reading must remain visible to users with a user_saved_readings row.

-- readings table policy
DROP POLICY IF EXISTS "read_readings" ON public.readings;
CREATE POLICY "read_readings" ON public.readings
  FOR SELECT TO authenticated
  USING (
    -- Normal case: reading is live and public/unlisted or owned by the caller
    (
      is_deleted = false
      AND (
        visibility = ANY (ARRAY['public'::public.reading_visibility, 'unlisted'::public.reading_visibility])
        OR owner_id = auth.uid()
      )
    )
    OR
    -- Saved case: caller has saved this reading (survives soft-delete)
    EXISTS (
      SELECT 1 FROM public.user_saved_readings usr
      WHERE usr.reading_id = readings.id
        AND usr.user_id    = auth.uid()
    )
  );

-- storage objects policy (readings bucket)
DROP POLICY IF EXISTS "read_readings_policy 1koz6g7_0" ON storage.objects;
CREATE POLICY "read_readings_policy 1koz6g7_0"
  ON storage.objects
  AS permissive
  FOR SELECT
  TO authenticated
  USING (
    bucket_id = 'readings'
    AND name ~ '^[0-9a-fA-F-]{36}(\..*)?'
    AND EXISTS (
      SELECT 1 FROM public.readings r
      WHERE r.id = (substring(objects.name, '^([0-9a-fA-F-]{36})'))::uuid
        AND (
          -- Live reading: public/unlisted or owned
          (
            r.is_deleted = false
            AND (r.visibility = 'public'::public.reading_visibility OR r.owner_id = auth.uid())
          )
          OR
          -- Saved reading: caller saved it
          EXISTS (
            SELECT 1 FROM public.user_saved_readings usr
            WHERE usr.reading_id = r.id
              AND usr.user_id    = auth.uid()
          )
        )
    )
  );
