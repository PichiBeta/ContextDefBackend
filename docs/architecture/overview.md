# Architecture Overview

## Data Flow: Reading Creation

```
Client
  → create-reading (Edge Fn, Bearer token auth)
    → INSERT readings row (status: uploading)
    → Upload text to Storage: readings/{reading_id}
    → UPDATE status: uploaded
      → DB trigger: readings_enqueue_difficulty
        → pg_net HTTP call → process-reading (Edge Fn, webhook secret auth)
          → Download text from Storage
          → tokenizeReading()         pure fn → ReadingStructureV1
          → calculateDifficulty()     pure fn → 0–100 score
          → Upload structure JSON to Storage: readings/{reading_id}.structure.v1.json
          → calculateReadingEmbedding()  async fn → 384-dim vector
          → UPDATE readings: difficulty, embedding, status: processed
```

## Database Schema

### Tables

- **`profiles`** — User profile. Has `embedding vector(384)` + `num_vectors int` for running-average user embedding. Auto-created by `handle_new_user` trigger on auth signup.
- **`readings`** — Text readings. Status lifecycle: `uploading → uploaded → processing → processed` (or `failed`). Has `difficulty int`, `embedding vector(384)`. `storage_path` is a GENERATED column (`id::text`). `content_preview varchar(70)` is NOT NULL.
- **`user_saved_readings`** — Junction table: users ↔ readings library. Has state: `active | pinned | archived`. User column is `user_id`.
- **`temp_readings`** — Prototype table for testing the difficulty pipeline. Ignore for new features.

### Key Functions and Triggers

- **`readings_enqueue_difficulty`** — Trigger that fires when `readings.status` becomes `'uploaded'`. Reads `READINGS_DIFFICULTY_WEBHOOK_SECRET` from Vault and calls `process-reading` via `pg_net` (async, fire-and-forget). Webhook URL is hardcoded to the production project URL.
- **`handle_new_user`** — Trigger that auto-creates a `profiles` row on auth signup.
- **`compute_new_embedding(old, new, n)`** — SQL function for incremental embedding average. Called by `calculate_user_embedding`.
- **`handle_updated_at`** — Trigger that auto-updates `updated_at` on all tables that have it.

### Extensions

`pgvector` (vector search), `pg_net` (HTTP webhooks), `pg_graphql`, `supabase_vault`.

For per-function details (inventory, auth patterns, secrets), see [docs/edge-functions/overview.md](../edge-functions/overview.md).
