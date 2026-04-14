-- Fixes to soft_delete_reading applied after initial deploy:
-- 1. Set visibility = 'private' to satisfy the readings_public_not_deleted CHECK constraint
--    (public readings cannot have is_deleted = true).
-- 2. Delete the owner's own user_saved_readings entry on soft-delete.
--    Other users' saved entries are preserved so they retain access.

CREATE OR REPLACE FUNCTION public.soft_delete_reading(reading_id uuid)
RETURNS void
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path TO 'pg_catalog', 'public'
AS $$
BEGIN
  UPDATE public.readings
  SET
    is_deleted = true,
    deleted_at = now(),
    status     = 'deleted',
    visibility = 'private'
  WHERE id         = reading_id
    AND owner_id   = auth.uid()
    AND is_deleted = false;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'reading not found or not owned by current user'
      USING ERRCODE = 'P0002';
  END IF;

  -- Remove the owner's own saved entry (if any).
  -- Other users' saved entries are preserved so they retain access.
  DELETE FROM public.user_saved_readings
  WHERE reading_id = soft_delete_reading.reading_id
    AND user_id    = auth.uid();
END;
$$;

GRANT EXECUTE ON FUNCTION public.soft_delete_reading(uuid) TO authenticated;
