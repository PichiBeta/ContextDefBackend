# Decision Log: personal-feed edge function (2026-04-09)

## What was built

New client-facing edge function `personal-feed` and its backing database RPC
`get_personal_feed`. Returns a ranked list of public readings for an
authenticated user, personalised by vector similarity when a user embedding
exists, or falling back to a recency-ordered discovery feed for new users.

## Key decisions

### 1. Two-branch PL/pgSQL RPC instead of a single CASE expression

The HNSW index on `readings.embedding` (`idx_readings_embedding`) is only
utilised by the Postgres planner when the ORDER BY clause is a plain
`embedding <=> query_vector` expression followed by LIMIT. A CASE expression in
ORDER BY prevents this optimisation. The function therefore uses two separate
`RETURN QUERY` branches — one for the personalized path (distance ORDER BY) and
one for the discovery path (recency ORDER BY) — so each path can use its
dedicated index.

### 2. SECURITY INVOKER + auth.uid() for saved-reading exclusion

Rather than accepting a `p_user_id uuid` parameter (which a caller could spoof),
the function uses `SECURITY INVOKER` so that `auth.uid()` resolves to the JWT of
the calling user. This removes a class of authorisation bugs where one user could
request the exclusion list of another.

### 3. Exclude already-saved readings

Readings already in the user's library are excluded from both feed modes. This
avoids showing users content they already own and keeps the feed useful as a
discovery surface.

### 4. feed_type field in response

The response includes `feed_type: "personalized" | "discovery"` so the frontend
can show different UI affordances (e.g. "Based on your reading history" vs
"Explore new readings") without needing to inspect individual fields.

### 5. similarity is null in discovery mode

Rather than omitting the field entirely (which could break client destructuring),
`similarity` is always present in each row but is `null` when the discovery path
is used. The edge function surfaces this as-is.

## Suggested commit message

```
feat: add personal-feed edge function with personalized/discovery modes

Adds get_personal_feed RPC and personal-feed edge function. Returns public
readings ranked by cosine similarity for users with embeddings, falling back
to recency for new users. Already-saved readings are excluded from both modes.
```
