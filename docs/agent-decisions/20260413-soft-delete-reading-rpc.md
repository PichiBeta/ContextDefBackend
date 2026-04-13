# 2026-04-13 — soft_delete_reading RPC

## Problem

The frontend was calling an RPC that set only `is_deleted = true` on a reading. This failed because the `readings_deleted_consistency` CHECK constraint requires all three fields to change atomically:

```sql
CONSTRAINT "readings_deleted_consistency" CHECK (
  (is_deleted = true  AND deleted_at IS NOT NULL AND status = 'deleted')
  OR
  (is_deleted = false AND deleted_at IS NULL     AND status <> 'deleted')
)
```

A secondary issue: the `update_own_readings` RLS USING clause filters on `is_deleted = false`, so any row already soft-deleted would also be blocked from being updated directly.

## Decision

Added a `soft_delete_reading(reading_id uuid)` RPC (`SECURITY INVOKER`) in migration `20260413000000_soft-delete-reading-rpc.sql`.

The function sets `is_deleted`, `deleted_at`, and `status` together in one `UPDATE`, with an ownership + `is_deleted = false` guard in the `WHERE` clause. It raises `P0002` (no_data_found) if the row doesn't exist or isn't owned by the calling user.

`GRANT EXECUTE … TO authenticated` is included so the frontend Supabase client can call it directly.

## Frontend call

```ts
await supabase.rpc('soft_delete_reading', { reading_id: id })
```
