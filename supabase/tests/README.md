# Test Strategy

## Unit Tests — no Docker required

Tests for pure, I/O-free modules. Run from the repo root:

```bash
cd supabase && deno test tests/unit/ --allow-read
```

Currently covers:
- `process-reading/calculate-difficulty.ts` — Flesch-Kincaid scoring
- `process-reading/tokenize-reading.ts` — tokenization, sentence/paragraph spans, codepoint offsets

These tests import the modules directly and require no network access or local Supabase stack.

## Integration Tests — requires `supabase start`

Tests that exercise the full local stack (DB, Storage, Edge Functions via HTTP).

### Setup

1. Start Docker Desktop
2. `supabase start`
3. In a second terminal: `supabase functions serve`
4. Run tests:

```bash
cd supabase && deno test tests/integration/ --allow-net --allow-env
```

### Secrets for integration tests

Integration tests read from `supabase/.env.local`. Provide at minimum:
- `SUPABASE_URL` (default: `http://127.0.0.1:54321`)
- `SUPABASE_SERVICE_ROLE_KEY` (shown by `supabase status`)
- `SUPABASE_ANON_KEY` (shown by `supabase status`)
- `READINGS_DIFFICULTY_WEBHOOK_SECRET` (any string for local testing)
- `HF_API_KEY` (required by `process-reading` — use a real key or mock)

### Test isolation

Each integration test creates its own test user and data, and cleans up in `afterEach`.
Do not rely on pre-existing data in the local DB.
