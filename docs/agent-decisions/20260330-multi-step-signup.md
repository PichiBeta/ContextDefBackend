# Multi-step sign-up with OTP email verification

**Date:** 2026-03-30

## Decision

Split the sign-up flow into multiple steps and enable OTP-based email verification. The `handle_new_user()` trigger now creates a skeleton profile (id, full_name, avatar_url only). Username and language preferences are collected on a separate onboarding screen and written via UPDATE after the user verifies their email with a 6-digit OTP code.

## Rationale

- Prepares the backend for future OAuth providers (e.g., Google), which don't carry custom metadata like `username`, `native_language`, or `target_language` in `raw_user_meta_data`.
- OTP-based verification avoids redirect URL complexity, which is problematic for mobile apps in developer mode and for cross-platform (app + web) support.
- `verifyOtp({ email, token, type: 'signup' })` both verifies the email and returns a session in one call, enabling auto-login after verification.

## What changed

### Migration (`20260330220242_multi-step-signup.sql`)

- `CREATE OR REPLACE FUNCTION public.handle_new_user()` — simplified to only insert `id`, `full_name`, `avatar_url` into `profiles`. No longer extracts `username`, `native_language`, or `target_language` from `raw_user_meta_data`.

### Config (`supabase/config.toml`)

- `enable_confirmations` set to `true` under `[auth.email]` — users must verify their email via OTP before signing in.

### No schema or RLS changes

- `username`, `native_language`, `target_language` are already nullable on `profiles`.
- CHECK constraints (`different_languages`, `username_length`) only apply to non-null values.
- Existing RLS policy `"Users can update own profile."` permits the post-verification UPDATE.

## Frontend flow (for reference)

```
Auth screen (email + password) → signUp() → skeleton profile created, OTP sent
Metadata screen → collect username, languages → store in client state
Verify screen → enter 6-digit OTP → verifyOtp() → session returned
Auto → UPDATE profiles with metadata → navigate to app
```

## Manual production steps

- Update the "Confirm signup" email template in Supabase dashboard to include `{{ .Token }}` (the OTP code).
