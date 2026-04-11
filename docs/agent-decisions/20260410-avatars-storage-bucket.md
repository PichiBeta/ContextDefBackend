# Decision Log: avatars storage bucket (2026-04-10)

## What was built

New `avatars` storage bucket with four RLS policies on `storage.objects`, created
via `supabase/migrations/20260411013702_create-avatars-bucket.sql`. Avatar files
are stored at `{userId}/avatar.{ext}` (e.g. `abc-123/avatar.jpg`). The public URL
is stored in `profiles.avatar_url` — no schema change to `profiles` was needed.

## Key decisions

### 1. Public bucket

The bucket is `public = true` so avatar URLs are directly accessible as plain
`https://` links (used in `<img src>` tags). Auth tokens are not required to read
an image, which is the standard pattern for profile pictures.

### 2. SELECT restricted to `authenticated`, not `anon`

Although the bucket is public (files are URL-accessible without a token), the RLS
SELECT policy is scoped to `TO authenticated`. Anonymous users fetching the URL
directly bypass RLS entirely (public bucket behaviour), so this policy governs
Supabase client SDK calls only. Restricting to `authenticated` prevents the anon
role from being used to enumerate the bucket via the storage API.

### 3. Folder-based path enforced by `storage.foldername`

Files must be uploaded to `{userId}/avatar.{ext}`. The INSERT, UPDATE, and DELETE
policies all check `(storage.foldername(name))[1] = auth.uid()::text`, which
extracts the first path segment and compares it to the caller's JWT UID. This is
the built-in Supabase helper for this pattern and is consistent with how
`readings` bucket policies work elsewhere in the project.

### 4. UPDATE has both USING and WITH CHECK

PostgreSQL UPDATE policies apply `USING` to determine which rows are eligible and
`WITH CHECK` to validate the row's state after the update. Without `WITH CHECK`,
a user could move their file to another user's folder path (the pre-update check
would pass, but the post-update row would be owned by a different folder).
Both clauses contain the same uid check.

### 5. `CREATE OR REPLACE POLICY` for idempotency

Migrations in this project can be replayed (e.g. `supabase db reset` in local
dev). Using `CREATE OR REPLACE POLICY` (available in PG 15, which Supabase uses)
makes the policy statements safe to re-run without a preceding `DROP`.

### 6. `image/*` wildcard and 10 MB limit

`image/*` covers all common formats (JPEG, PNG, WebP, GIF, AVIF, etc.) without
maintaining an explicit allowlist. 10 MB is large enough for any reasonable
high-resolution avatar while keeping the bucket from being used as general-purpose
image storage.

## Suggested commit message

```
feat: add avatars storage bucket with RLS policies
```
