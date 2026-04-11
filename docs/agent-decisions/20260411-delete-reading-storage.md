# Decision Log: delete-reading-storage (2026-04-11)

## What was built

New `delete-reading-storage` edge function, backed by a new `readings_enqueue_storage_cleanup` PL/pgSQL trigger function and `trg_readings_enqueue_storage_cleanup` trigger on `public.readings`.

When a reading row is hard-deleted from `public.readings`, the trigger fires asynchronously via `pg_net`, calling the edge function to delete all associated files in the `readings` Storage bucket.

**Files added:**
- `supabase/migrations/20260411120000_readings-storage-cleanup-trigger.sql`
- `supabase/functions/delete-reading-storage/index.ts`
- `supabase/functions/delete-reading-storage/deno.json`
- `docs/edge-functions/delete-reading-storage.md`

**Files modified:**
- `supabase/deno.json` — added workspace member
- `supabase/config.toml` — added `[functions.delete-reading-storage]` stanza

## Key decisions

### 1. AFTER DELETE trigger (not soft-delete)

The `readings` table uses soft-delete (`is_deleted`, `status = 'deleted'`). This trigger fires on **hard DELETE** of the row, not on the soft-delete transition. Soft deletion leaves the row (and the data) intact; hard deletion from the Supabase console is the intended cleanup event.

### 2. AFTER DELETE, not BEFORE DELETE

`AFTER DELETE` is correct because cleanup is a side effect of an already-committed deletion. Using `BEFORE DELETE` would block the delete transaction while the Vault lookup happens unnecessarily. `pg_net` fires HTTP requests after the transaction commits regardless of BEFORE/AFTER, so AFTER is semantically accurate and slightly cheaper.

### 3. Prefix-based list + batch delete

Rather than enumerating the two known file paths (`{id}` and `{id}.structure.v1.json`), the edge function calls `storage.list("", { search: reading_id })` and filters with `.startsWith(reading_id)`. This handles both known files and any future derivatives (e.g. audio, translated structures) without code changes.

### 4. Reuses READINGS_DIFFICULTY_WEBHOOK_SECRET

No new Vault secret was created. The overview doc confirms all backend webhook functions share this one secret (the name is historical). Adding a new secret would add operational burden for no security benefit.

### 5. Empty-list is success (idempotent)

If no files are found (e.g. the reading was deleted before its upload completed), the function returns `{ ok: true, deleted_paths: [] }`. This avoids spurious errors for a normal edge case.

### 6. No row update on failure

Unlike `process-reading`, which updates `readings.status = 'failed'` on exception, `delete-reading-storage` has no row to update — it was already hard-deleted. Failures are logged to console and return HTTP 500. Remaining files can be cleaned up manually via the Storage console.

## Suggested commit message

```
feat: auto-delete storage files when a reading is hard-deleted

Adds readings_enqueue_storage_cleanup trigger + delete-reading-storage
edge function. When a row is hard-deleted from public.readings, pg_net
fires the edge function async; it prefix-lists and batch-removes all
files for that reading in the readings Storage bucket.
```
