# Remove `temp_readings` table and `calculate-difficulty` edge function

**Date:** 2026-03-26

## Decision

Remove the `temp_readings` prototype table, its triggers/functions, and the standalone `calculate-difficulty` edge function.

## Rationale

- `temp_readings` was a prototype table for testing the difficulty pipeline. The production pipeline runs through the `readings` table and `process-reading` edge function.
- `calculate-difficulty` (the edge function) contained logic identical to `process-reading/calculate-difficulty.ts`. Its only caller was the `temp_readings_enqueue_difficulty` trigger on the prototype table.
- With `temp_readings` removed, the standalone edge function has no consumers.

## What was removed

### Migration (`20260326235138_remove-temp-readings-and-calculate-difficulty.sql`)

Drops in dependency order:
1. Triggers: `trg_temp_readings_enqueue_difficulty`, `trg_temp_readings_set_updated_at`
2. RLS policy: `"testing readings"`
3. Function: `temp_readings_enqueue_difficulty()`
4. Table: `temp_readings`

### Files deleted

- `supabase/functions/calculate-difficulty/` (entire directory)
- `docs/edge-functions/calculate-difficulty.md` (stub doc)

### Files updated

- `supabase/config.toml` — removed `[functions.calculate-difficulty]` block
- `supabase/deno.json` — removed workspace entry
- `supabase/.env.example` — removed `TEMP_READINGS_DIFFICULTY_WEBHOOK_SECRET`
- `docs/edge-functions/overview.md` — removed `calculate-difficulty` from tables
- `docs/architecture/overview.md` — removed `temp_readings` mention
- `CLAUDE.md` — removed resolved tech debt entries

## What was kept

- `supabase/functions/process-reading/calculate-difficulty.ts` — the canonical difficulty calculation module
- `supabase/tests/unit/calculate-difficulty.test.ts` — tests the process-reading module, not the deleted edge function
- `READINGS_DIFFICULTY_WEBHOOK_SECRET` — still used by `process-reading`

## Post-deploy manual steps

- Remove `TEMP_READINGS_DIFFICULTY_WEBHOOK_SECRET` from Dashboard > Vault
- Run `supabase functions delete calculate-difficulty --project-ref <ref>`

## Suggested commit message

`Remove temp_readings table and calculate-difficulty edge function`
