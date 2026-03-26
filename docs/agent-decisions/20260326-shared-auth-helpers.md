# 2026-03-26 — Shared Auth Helpers

## What changed

Created `functions/_shared/auth.ts` with two exported helpers:

- `requireUser(req)` — validates Bearer JWT, returns `{ user, supabase }` or a 401 Response
- `requireWebhookSecret(req, secret)` — validates `x-webhook-secret` header, returns `true` or a 401 Response

Updated `create-reading`, `defintion-translation`, `process-reading`, and `calculate-difficulty` to use these helpers.

## Why

Two inconsistencies existed between the JWT-auth functions:

1. `defintion-translation` used `SUPABASE_ANON_KEY` instead of `SUPABASE_PUBLISHABLE_KEY` — a bug since this project uses the publishable key pattern.
2. Error response format differed: `{ error: "..." }` vs `{ ok: false, error: "..." }`. All 401s now use `{ ok: false, error: "Unauthorized" }`.

The webhook functions were already consistent with each other, but abstracting them standardizes the response format and keeps auth logic in one place.

## What was not changed

- `ocr-extract` and `calculate_user_embedding` still have no auth — tracked as tech debt in CLAUDE.md.
- No changes to auth behavior for any function — only the implementation location changed.
