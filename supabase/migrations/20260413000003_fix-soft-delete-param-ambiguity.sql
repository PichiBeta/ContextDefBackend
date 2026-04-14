-- Fixes "column reference reading_id is ambiguous" in soft_delete_reading.
-- The parameter name conflicted with the reading_id column in user_saved_readings
-- inside the same function body. Renaming the parameter to p_reading_id removes
-- the ambiguity without changing the function's behaviour or signature.

CREATE OR REPLACE FUNCTION public.soft_delete_reading(p_reading_id uuid)
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
  WHERE id         = p_reading_id
    AND owner_id   = auth.uid()
    AND is_deleted = false;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'reading not found or not owned by current user'
      USING ERRCODE = 'P0002';
  END IF;

  -- Remove the owner's own saved entry (if any).
  -- Other users' saved entries are preserved so they retain access.
  DELETE FROM public.user_saved_readings
  WHERE reading_id = p_reading_id
    AND user_id    = auth.uid();
END;
$$;

GRANT EXECUTE ON FUNCTION public.soft_delete_reading(uuid) TO authenticated;
