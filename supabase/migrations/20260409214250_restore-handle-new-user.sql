-- Restore handle_new_user() to populate all profile fields from raw_user_meta_data.
-- The previous migration (20260330220242) simplified this to skeleton-only (id, full_name, avatar_url).
-- The frontend now sends username, full_name, native_language, and target_language as options.data
-- in the signUp() call, so all fields are available in raw_user_meta_data at account creation time.
--
-- Also tightens the profiles schema: full_name, username, native_language, and target_language are
-- now NOT NULL since they are always provided at sign-up. Run this only after ensuring no existing
-- rows have NULLs in these columns (delete any skeleton/test profiles first if needed).

CREATE OR REPLACE FUNCTION public.handle_new_user() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO ''
    AS $$
begin
  insert into public.profiles (id, full_name, avatar_url, username, native_language, target_language)
  values (
    new.id,
    new.raw_user_meta_data->>'full_name',
    new.raw_user_meta_data->>'avatar_url',
    new.raw_user_meta_data->>'username',
    (new.raw_user_meta_data->>'native_language')::public.language_code,
    (new.raw_user_meta_data->>'target_language')::public.language_code
  );
  return new;
end;
$$;

ALTER TABLE public.profiles
    ALTER COLUMN full_name SET NOT NULL,
    ALTER COLUMN username SET NOT NULL,
    ALTER COLUMN native_language SET NOT NULL,
    ALTER COLUMN target_language SET NOT NULL;
