set check_function_bodies = off;

CREATE OR REPLACE FUNCTION public.handle_new_user()
 RETURNS trigger
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO ''
AS $function$
begin
  insert into public.profiles (id, full_name, avatar_url)
  values (
    new.id,
    new.raw_user_meta_data->>'full_name',
    new.raw_user_meta_data->>'avatar_url'
  );
  return new;
end;
$function$
;

CREATE OR REPLACE FUNCTION public.handle_updated_at()
 RETURNS trigger
 LANGUAGE plpgsql
AS $function$
begin
  new.updated_at = now();
  return new;
end;
$function$
;

CREATE OR REPLACE FUNCTION public.readings_enqueue_difficulty()
 RETURNS trigger
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$declare
  _url text := 'https://irspwhgeyrojqluzgciu.supabase.co/functions/v1/process-reading';
  _secret text;
  _payload jsonb;
begin
  -- Only act on a transition to 'uploaded'
  if tg_op <> 'UPDATE'
     or new.status <> 'uploaded'
     or old.status is not distinct from new.status
     or new.is_deleted = true
  then
    return new;
  end if;

  -- Mark processing synchronously inside the same transaction.
  new.status := 'processing';
  new.difficulty := null;
  new.error_message := null;

  -- Pull secret from Vault
  select decrypted_secret
    into _secret
  from vault.decrypted_secrets
  where name = 'READINGS_DIFFICULTY_WEBHOOK_SECRET'
  limit 1;

  if _secret is null then
    raise exception 'Vault secret "%" not found', 'READINGS_DIFFICULTY_WEBHOOK_SECRET';
  end if;

  _payload := jsonb_build_object(
    'reading_id', new.id,
    'language_code', new.language_code,
    'storage_path', (new.id::text),
    'content_updated_at', new.content_updated_at
  );

  -- Fire and forget (async). Actual HTTP starts after commit.
  perform net.http_post(
    url := _url,
    headers := jsonb_build_object(
      'content-type', 'application/json',
      'x-webhook-secret', _secret
    ),
    body := _payload
  );

  return new;
end;$function$
;

drop policy "delete_own_readings_object 1koz6g7_0" on "storage"."objects";

drop policy "insert_own_readings_object 1koz6g7_0" on "storage"."objects";

drop policy "read_readings_policy 1koz6g7_0" on "storage"."objects";


  create policy "delete_own_readings_object 1koz6g7_0"
  on "storage"."objects"
  as permissive
  for delete
  to authenticated
using (((bucket_id = 'readings'::text) AND (name ~ '^[0-9a-fA-F-]{36}(\..*)?;
::text) AND (EXISTS ( SELECT 1
   FROM public.readings r
  WHERE ((r.id = ("substring"(objects.name, '^([0-9a-fA-F-]{36})'::text))::uuid) AND (r.is_deleted = false) AND (r.owner_id = auth.uid()))))));



  create policy "insert_own_readings_object 1koz6g7_0"
  on "storage"."objects"
  as permissive
  for insert
  to authenticated
with check (((bucket_id = 'readings'::text) AND (name ~ '^[0-9a-fA-F-]{36}(\..*)?;
::text) AND (EXISTS ( SELECT 1
   FROM public.readings r
  WHERE ((r.id = ("substring"(objects.name, '^([0-9a-fA-F-]{36})'::text))::uuid) AND (r.is_deleted = false) AND (r.owner_id = auth.uid()) AND (r.status = 'uploading'::public.reading_status))))));



  create policy "read_readings_policy 1koz6g7_0"
  on "storage"."objects"
  as permissive
  for select
  to authenticated
using (((bucket_id = 'readings'::text) AND (name ~ '^[0-9a-fA-F-]{36}(\..*)?;
::text) AND (EXISTS ( SELECT 1
   FROM public.readings r
  WHERE ((r.id = ("substring"(objects.name, '^([0-9a-fA-F-]{36})'::text))::uuid) AND (r.is_deleted = false) AND ((r.visibility = 'public'::public.reading_visibility) OR (r.owner_id = auth.uid())))))));



