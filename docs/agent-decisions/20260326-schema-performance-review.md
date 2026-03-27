# Schema performance review and code-first alignment

**Date:** 2026-03-26

## Decision

Address four issues found during a full production-vs-migrations audit:

1. **Pull `words_lookup_cache` and `words_saved` into migrations** — both were created on the dashboard before the code-first policy. Migration uses `IF NOT EXISTS` guards so it no-ops on production.
2. **Add HNSW vector indexes** on `profiles.embedding` and `readings.embedding` — required for any future similarity search to avoid full table scans.
3. **Tighten `profiles` SELECT RLS** from all roles to `authenticated` only — prevents unauthenticated access to profile data.
4. **Change `readings.owner_id` FK to ON DELETE SET NULL** — preserves readings when a profile is deleted, consistent with the soft-delete pattern.

## Rationale

- The two missing tables were a code-first compliance gap. `supabase db pull` did not capture them (likely created after the initial pull).
- Embedding columns had the `vector` extension and type but no indexes, making similarity queries O(n).
- The profiles SELECT policy was `TO ALL USING (true)`, exposing embeddings and language preferences to unauthenticated requests.
- The `readings.owner_id` FK defaulted to `NO ACTION`, which would block profile deletion if the user owned any readings. SET NULL was chosen over CASCADE to preserve public content.

## Migrations

- `20260327005318_pull-dashboard-tables.sql` — `words_lookup_cache` + `words_saved` (tables, indexes, RLS, triggers)
- `20260327005322_schema-performance-fixes.sql` — HNSW indexes, profiles RLS, owner_id FK
