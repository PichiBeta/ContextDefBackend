-- Soft-delete RPC for readings.
-- The readings table has a readings_deleted_consistency CHECK constraint that
-- requires is_deleted, deleted_at, and status to change together atomically.
-- A bare UPDATE setting only is_deleted = true violates that constraint.
-- This function sets all three fields in one statement and enforces ownership.
-- user_saved_readings rows are intentionally preserved: users who saved a reading
-- retain access to it even after the owner soft-deletes it (see read_readings policy).

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
    status     = 'deleted'
  WHERE id        = reading_id
    AND owner_id  = auth.uid()
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
