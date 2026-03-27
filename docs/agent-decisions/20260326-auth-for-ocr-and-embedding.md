# Add auth to ocr-extract and calculate_user_embedding

**Date:** 2026-03-26

## Decision

Closed the auth gap on the two remaining unauthenticated edge functions:

- **`ocr-extract`** (client-facing) — added `requireUser(req)` to verify the caller's JWT, consistent with `create-reading` and `defintion-translation`.
- **`calculate_user_embedding`** (backend-called) — added `requireWebhookSecret(req, secret)` using the existing `READINGS_DIFFICULTY_WEBHOOK_SECRET`, consistent with `process-reading` and `calculate-difficulty`. All backend-invoked functions share this single secret (the name is historical).

## Action required

The backend (FastAPI) caller must send the `x-webhook-secret` header with the value of `READINGS_DIFFICULTY_WEBHOOK_SECRET` when calling `calculate_user_embedding`.

## Suggested commit message

`Add auth to ocr-extract (requireUser) and calculate_user_embedding (requireWebhookSecret)`
