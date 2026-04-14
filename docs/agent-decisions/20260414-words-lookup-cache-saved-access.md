# words_lookup_cache & words_saved — Saved-Reading RLS Access

**Date:** 2026-04-14  
**Migration:** `20260414000000_words-lookup-cache-saved-access.sql`

## Problem

After soft-delete was introduced for readings (`20260413000001_read-readings-saved-access.sql`), users who have a reading in their library retain access to the reading itself. However, three policies on `words_lookup_cache` and two policies on `words_saved` still gated access on `is_deleted = false`, so saved-reading users lost the ability to:

- Look up word definitions (SELECT/INSERT/UPDATE on `words_lookup_cache`)
- Save or update vocabulary entries (INSERT/UPDATE on `words_saved`)

for any reading that had been soft-deleted.

## Decision

Add the same saved-reading OR branch already used in the `read_readings` policy:

```sql
OR EXISTS (
    SELECT 1 FROM public.user_saved_readings usr
    WHERE usr.reading_id = r.id
      AND usr.user_id = auth.uid()
)
```

This was applied to:
- `words_lookup_cache`: SELECT, INSERT, UPDATE policies
- `words_saved`: INSERT, UPDATE policies (SELECT and DELETE only check `user_id = auth.uid()`, no reading-availability check needed)

## Alternatives Considered

**Allow any authenticated user** — Rejected. This would expose cache data for private or soft-deleted readings to users with no relationship to the reading. Access should be gated on an actual relationship (owner or saver).

## Commit Message Suggestion

```
fix: extend words_lookup_cache and words_saved RLS to allow saved-reading access after soft-delete
```
