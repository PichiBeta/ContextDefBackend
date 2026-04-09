# Edge Function: `personal-feed`

## Purpose

Returns a ranked list of public, processed readings for the authenticated user.

## Auth

Bearer token (`requireUser`) — standard client-facing pattern.

## Method & Endpoint

```
GET /functions/v1/personal-feed
Authorization: Bearer <jwt>
```

### Query Parameters

| Param | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `limit` | int | 20 | 1–100 | Number of readings to return |
| `offset` | int | 0 | ≥0 | Pagination offset |

## Behaviour

1. Fetches `profiles.embedding` and `profiles.num_vectors` for the caller.
2. If `num_vectors > 0` (user has saved at least one reading):
   - Calls `get_personal_feed` RPC with the user's embedding.
   - Results are ordered by **cosine similarity** (highest first) via the HNSW index on `readings.embedding`.
   - `feed_type` in response is `"personalized"`.
3. If `num_vectors === 0` (no saved readings yet):
   - Calls `get_personal_feed` RPC with `p_user_embedding = null`.
   - Results are ordered by **recency** (`created_at DESC`).
   - `feed_type` in response is `"discovery"`.

Readings already saved by the user are **excluded** from both modes.

## Response

### Success (200)

```json
{
  "ok": true,
  "feed_type": "personalized",
  "feed": [
    {
      "id": "uuid",
      "title": "string",
      "genre": "string",
      "language_code": "en",
      "content_preview": "string (≤70 chars)",
      "difficulty": 45,
      "owner_id": "uuid",
      "created_at": "2026-04-01T00:00:00Z",
      "similarity": 0.87
    }
  ]
}
```

- `similarity` is a float in [0, 1] (1 = identical). It is `null` in discovery mode.
- `feed_type` is `"personalized"` or `"discovery"`.

### Errors

| Status | Body | Cause |
|---|---|---|
| 401 | `{ ok: false, error: "Unauthorized" }` | Missing or invalid Bearer token |
| 405 | `{ ok: false, error: "Method not allowed" }` | Non-GET request |
| 500 | `{ ok: false, error: "...", detail: "..." }` | Profile fetch or RPC failure |

## Database RPC

`public.get_personal_feed(p_user_embedding vector(384), p_limit int, p_offset int)`

- `SECURITY INVOKER` — `auth.uid()` resolves to the caller's JWT for the `NOT EXISTS` exclusion subquery.
- Personalized path uses `idx_readings_embedding` (HNSW, cosine).
- Discovery path uses `idx_readings_feed_created_at` (btree, partial).
- Granted to `authenticated` role only.

## Usage

The user identity is derived from the JWT — no user ID is passed explicitly.

```ts
const { data: { session } } = await supabase.auth.getSession();

const response = await fetch(
  `${SUPABASE_URL}/functions/v1/personal-feed?limit=20&offset=0`,
  {
    method: "GET",
    headers: {
      "Authorization": `Bearer ${session.access_token}`,
      "apikey": SUPABASE_PUBLISHABLE_KEY,
    },
  }
);

const data = await response.json();
// data.feed_type → "personalized" | "discovery"
// data.feed      → Reading[]
```

### Pagination

Increment `offset` by `limit` on each subsequent page. Fire the next request
when the user has scrolled through ~70–80% of the currently loaded items.
Stop fetching when the returned `feed` array is shorter than `limit`.

```ts
// page 1
?limit=20&offset=0

// page 2
?limit=20&offset=20

// page 3
?limit=20&offset=40
```

Deduplicate by `id` on the frontend before rendering, in case a ranking shift
causes the same reading to appear across two pages.

## Secrets

None beyond the standard `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY` (inherited from the shared auth helper).
