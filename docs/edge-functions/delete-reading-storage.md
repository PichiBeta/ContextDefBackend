# delete-reading-storage

## Purpose

Cleans up Supabase Storage files in the `readings` bucket when a reading row is hard-deleted from `public.readings` (e.g. via the Supabase console). Called asynchronously by the `trg_readings_enqueue_storage_cleanup` DB trigger via `pg_net`.

## Auth

Webhook — `x-webhook-secret` header validated against `READINGS_DIFFICULTY_WEBHOOK_SECRET` (Vault secret, shared with `process-reading` and `calculate_user_embedding`).

## Payload

```json
{ "reading_id": "<uuid>" }
```

## Steps

| Step | Description |
|------|-------------|
| 1 | Validate `x-webhook-secret` header |
| 2 | Parse `reading_id` from JSON body |
| 3 | List all objects in the `readings` bucket with `reading_id` as prefix |
| 4 | Batch-delete matched paths (`storage.remove(paths)`) |
| 5 | Return `{ ok: true, deleted_paths: string[] }` |

## Known files per reading

| Path | Description |
|------|-------------|
| `{reading_id}` | Raw text file (no extension) |
| `{reading_id}.structure.v1.json` | Processed structure JSON |

Using a prefix-based list rather than enumerating paths explicitly means future file types are handled automatically.

## Failure behaviour

If the delete fails, the function returns HTTP 500 and logs the error. `pg_net` does not retry on non-2xx responses. Files will remain in Storage until manually cleaned up. This is an acceptable trade-off for an async best-effort cleanup path — the row is already gone, so there is no status field to update.

## Secrets required

| Secret | Source |
|--------|--------|
| `SUPABASE_URL` | Injected automatically by Supabase runtime |
| `SUPABASE_SERVICE_ROLE_KEY` | Injected automatically by Supabase runtime |
| `READINGS_DIFFICULTY_WEBHOOK_SECRET` | Vault (shared with `process-reading`) |
