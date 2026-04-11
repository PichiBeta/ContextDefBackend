-- Trigger + function to async-delete Storage files when a reading row is
-- hard-deleted from public.readings (e.g. via the Supabase console).
--
-- Flow: AFTER DELETE trigger → readings_enqueue_storage_cleanup()
--       → Vault secret lookup → net.http_post (fire-and-forget)
--       → delete-reading-storage edge function cleans up the bucket.

CREATE OR REPLACE FUNCTION public.readings_enqueue_storage_cleanup()
  RETURNS trigger
  LANGUAGE plpgsql
  SECURITY DEFINER
  SET search_path TO 'pg_catalog', 'public'
AS $function$
declare
  _url    text := 'https://irspwhgeyrojqluzgciu.supabase.co/functions/v1/delete-reading-storage';
  _secret text;
  _payload jsonb;
begin
  select decrypted_secret
    into _secret
  from vault.decrypted_secrets
  where name = 'READINGS_DIFFICULTY_WEBHOOK_SECRET'
  limit 1;

  if _secret is null then
    raise exception 'Vault secret "%" not found', 'READINGS_DIFFICULTY_WEBHOOK_SECRET';
  end if;

  _payload := jsonb_build_object('reading_id', old.id);

  -- Fire and forget (async). HTTP call starts after the transaction commits.
  perform net.http_post(
    url     := _url,
    headers := jsonb_build_object(
      'content-type',     'application/json',
      'x-webhook-secret', _secret
    ),
    body    := _payload
  );

  return old;
end;
$function$;

DROP TRIGGER IF EXISTS trg_readings_enqueue_storage_cleanup ON public.readings;
CREATE TRIGGER trg_readings_enqueue_storage_cleanup
  AFTER DELETE ON public.readings
  FOR EACH ROW
  EXECUTE FUNCTION public.readings_enqueue_storage_cleanup();

REVOKE ALL ON FUNCTION public.readings_enqueue_storage_cleanup() FROM PUBLIC;
GRANT ALL ON FUNCTION public.readings_enqueue_storage_cleanup() TO service_role;
