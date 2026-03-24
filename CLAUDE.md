# ContextDefBackend — Claude Code Context

## What This Repo Is

Backend for a language-learning reading platform. Users upload text readings in multiple languages; the system tokenizes them, scores their difficulty, and computes semantic embeddings for personalized recommendations. The primary concern of this repo is the Supabase project (database + edge functions). A small FastAPI service (`app/`) handles text chunking.

## Architecture Overview

### Repo Structure

```
supabase/
  config.toml               ← local dev config (Deno version, ports, per-function settings)
  migrations/               ← all schema changes as SQL files — source of truth for DB
  functions/                ← edge functions (TypeScript/Deno 2)
    process-reading/        ← main processing orchestrator (webhook)
    create-reading/         ← client-facing API
    calculate-difficulty/   ← webhook for temp_readings prototype table
    ocr-extract/            ← Azure OCR
    defintion-translation/  ← Claude AI word definitions + translation
    calculate_user_embedding/ ← user profile embedding updater
app/                        ← FastAPI (Python) text chunking service
scripts/
  sync-supabase.ps1         ← pulls latest DB schema + function code from dashboard
```

### Data Flow: Reading Creation

```
Client
  → create-reading (Edge Fn, JWT auth)
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

### Edge Functions Inventory

| Function | Type | Auth | Key Secrets |
|---|---|---|---|
| `process-reading` | Webhook (DB trigger) | `x-webhook-secret` header | `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `READINGS_DIFFICULTY_WEBHOOK_SECRET`, `HF_API_KEY` |
| `create-reading` | API (client) | JWT Bearer token | `SUPABASE_URL`, `SUPABASE_ANON_KEY` |
| `calculate-difficulty` | Webhook (DB trigger) | `x-webhook-secret` header | `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `READINGS_DIFFICULTY_WEBHOOK_SECRET` |
| `ocr-extract` | API (client) | None | `AZURE_DOC_INTEL_ENDPOINT`, `AZURE_DOC_INTEL_KEY` |
| `defintion-translation` | API (client) | None | `ANTHROPIC_API_KEY` |
| `calculate_user_embedding` | Internal API | None (service role) | `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` |

All functions use `verify_jwt = false` (see `[functions.*]` sections in `config.toml`).

### Database Schema (key objects)

**Tables:**
- `profiles` — User profile. Has `embedding vector(384)` + `num_vectors int` for running-average. Auto-created by `handle_new_user` trigger on auth signup.
- `readings` — Text readings. Status lifecycle: `uploading → uploaded → processing → processed` (or `failed`). Has `difficulty int`, `embedding vector(384)`.
- `user_saved_readings` — Junction: users ↔ readings library. Has state: `active | pinned | archived`.
- `temp_readings` — Prototype table for testing the difficulty pipeline. Ignore for new features.

**Key functions/triggers:**
- `readings_enqueue_difficulty` trigger — fires when `readings.status` becomes `'uploaded'`. Reads `READINGS_DIFFICULTY_WEBHOOK_SECRET` from Vault and calls `process-reading` via `pg_net` (async, fire-and-forget). Webhook URL is hardcoded to the production project URL.
- `handle_new_user` trigger — auto-creates `profiles` row on auth signup.
- `compute_new_embedding(old, new, n)` — SQL function for incremental embedding average. Called by `calculate_user_embedding`.
- `handle_updated_at` trigger — auto-updates `updated_at` on all tables that have it.

**Extensions:** `pgvector` (vector search), `pg_net` (HTTP webhooks), `pg_graphql`, `supabase_vault`.

### process-reading Internal Modules

The `process-reading/` function contains pure, I/O-free modules — ideal for unit testing:

| File | Type | What it does |
|---|---|---|
| `index.ts` | Orchestrator | Webhook auth, Storage download/upload, DB update |
| `tokenize-reading.ts` | Pure fn | `tokenizeReading(ctx)` → `ReadingStructureV1` (paragraphs, sentences, tokens with codepoint offsets) |
| `calculate-difficulty.ts` | Pure fn | `calculateDifficulty(ctx)` → `{ score: number }` (Flesch-Kincaid Grade Level, normalized 0–100) |
| `calculate-reading-embedding.ts` | Async fn | `calculateReadingEmbedding(ctx)` → `{ embedding, chunks_processed }` — HuggingFace MiniLM-L12-v2 |

---

## Development Conventions

### Code-First Rule
**All schema changes go through `supabase/migrations/`.** Never make schema changes directly in the dashboard without immediately pulling them:
```bash
supabase db pull <migration_name>    # requires Docker
```

### Function Naming
- New functions: **kebab-case** (`my-new-function/`)
- `calculate_user_embedding` uses snake_case because it was dashboard-created. Keep as-is — renaming a deployed function changes its URL slug and requires re-wiring callers.

### deno.json Required
Every function directory must have a `deno.json`. Functions created via the dashboard may lack one — add it before running locally. The import alias `"supabase"` should map to `jsr:@supabase/supabase-js@2` for consistency.

### Imports
Deno 2, JSR imports (`jsr:`), or npm specifiers (`npm:`). No CDN imports from esm.sh or deno.land/x.

### No Smart-Task
The `smart-task` function was deleted — it was the old standalone embed-reading function, now replaced by `process-reading/calculate-reading-embedding.ts`.

---

## Working With Secrets

See `supabase/.env.example` for the full list with descriptions.

**Local dev:**
```bash
cp supabase/.env.example supabase/.env.local
# Fill in real values — ask a teammate
```
`supabase/.env.local` is gitignored. The local stack reads it automatically when you run `supabase functions serve`.

**Production:**
```bash
supabase secrets set KEY=value --project-ref irspwhgeyrojqluzgciu
```

**Vault secrets** (`READINGS_DIFFICULTY_WEBHOOK_SECRET`, `TEMP_READINGS_DIFFICULTY_WEBHOOK_SECRET`) are stored in the Supabase Vault and read by Postgres trigger code — set them via Dashboard > Vault or the CLI vault commands.

---

## Local Development

### Prerequisites
- [Supabase CLI](https://supabase.com/docs/guides/cli/getting-started) — `supabase --version`
- Docker Desktop (must be **running** before `supabase start`)
- Deno 2.x — `deno --version`

### Start Local Stack
```bash
supabase start                  # starts Docker containers (first run: ~2 min)
supabase functions serve        # serves all edge functions locally (separate terminal)
```

Local URLs:
- API: http://localhost:54321
- Studio: http://localhost:54323
- Functions: http://localhost:54321/functions/v1/

### Apply Schema Changes Locally
```bash
supabase db push                # apply pending migrations to local DB
```

### Re-Sync from Dashboard
Run when you want to pull dashboard changes into the repo:
```bash
powershell -ExecutionPolicy Bypass -File .\scripts\sync-supabase.ps1 -SyncSecretsTemplate
```
This pulls function source + DB schema. Review `git diff supabase/` before committing.
The script warns if any deployed function is missing a `[functions.<name>]` section in `config.toml`.

---

## Running Tests

```bash
# Unit tests — no Docker required (run from supabase/ directory)
cd supabase && deno test tests/unit/ --allow-read

# Integration tests — requires: supabase start + supabase functions serve
cd supabase && deno test tests/integration/ --allow-net --allow-env
```

See [supabase/tests/README.md](supabase/tests/README.md) for test strategy details.

---

## Known Issues / Tech Debt

| Issue | Location | Notes |
|---|---|---|
| `calculate-difficulty` duplicates logic | `functions/calculate-difficulty/index.ts` | Only used by the `temp_readings` prototype table trigger. Low priority to consolidate. |
| Typo in function name | `functions/defintion-translation/` | "defintion" not "definition". Do NOT rename — it is the deployed URL slug. |
| Hardcoded production URL in trigger | `migrations/*.sql` → `readings_enqueue_difficulty` | `pg_net` triggers require a hardcoded URL. Local dev does NOT fire this trigger against the local function. |
| Orphaned dashboard secrets | `supabase/.env.example` | `GEMINI_API_KEY` and `OPEN_API_KEY` exist in the dashboard but no function code references them. Investigate before removing. |
| Vault secrets not in migrations | Dashboard > Vault | `READINGS_DIFFICULTY_WEBHOOK_SECRET` and `TEMP_READINGS_DIFFICULTY_WEBHOOK_SECRET` must be set manually in the Vault — they are not seeded by any migration. |
| Storage buckets not in migrations | Dashboard > Storage | The `readings` bucket must be created manually. See `docs/onboarding.md`. |
