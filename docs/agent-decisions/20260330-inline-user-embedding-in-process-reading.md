# 2026-03-30 — Inline User Embedding Update in process-reading

## What changed

Added step 8 to `process-reading/index.ts`: after the reading is successfully marked `processed` (step 7), the function now also updates the reading owner's profile embedding inline.

- Added `updateUserEmbedding(supabase, owner_id, readingEmbedding)` helper above `Deno.serve`
- Widened step 7's `.select("id")` to `.select("id, owner_id")` to retrieve `owner_id` from the update result without an extra DB query
- Step 8 is best-effort: wrapped in `try/catch`; a failure logs an error but does not affect the reading's `status` or the HTTP response

## Why

`calculate_user_embedding` was previously called separately after processing completed. For the reading owner specifically, this required a separate API call and a redundant DB fetch of the reading embedding — data already in memory at the end of `process-reading`.

Moving the update inline for this path eliminates both the extra call and the re-fetch. `calculate_user_embedding` remains active and unchanged for all other cases (e.g. when a user saves an existing reading).

## Files modified

- `supabase/functions/process-reading/index.ts`
- `docs/edge-functions/process-reading.md` (added orchestration steps table with step 8)
- `docs/edge-functions/calculate-user-embedding.md` (documented behavior; noted the inline path in process-reading)