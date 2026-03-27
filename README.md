# ContextDefBackend

Backend for a language-learning reading platform. Users upload text readings; the system tokenizes them, scores difficulty, and computes semantic embeddings for personalized recommendations.

## Quick Links

- [Architecture & Development Guide](CLAUDE.md)
- [Engineer Onboarding](docs/onboarding.md)
- [Test Strategy](supabase/tests/README.md)

## Repo Structure

| Path | Description |
|---|---|
| `supabase/` | Supabase project: database migrations, edge functions (Deno/TypeScript) |
| `supabase/functions/` | Edge functions — one directory per function |
| `supabase/migrations/` | SQL migration files — source of truth for DB schema |
| `supabase/config.toml` | Local dev configuration (ports, Deno version, function settings) |
| `supabase/tests/` | Unit and integration tests |
| `app/` | FastAPI service (Python) for text chunking |
| `scripts/sync-supabase.ps1` | Pulls latest schema + function code from the Supabase dashboard |

## First-Time Setup

See [docs/onboarding.md](docs/onboarding.md) for detailed instructions.

**Quick start:**

```bash
# 1. Install: Supabase CLI, Docker Desktop, Deno 2.x
# 2. Authenticate
supabase login
supabase link --project-ref irspwhgeyrojqluzgciu

# 3. Set up secrets
cp supabase/.env.example supabase/.env.local
# Fill in supabase/.env.local with real values (ask a teammate)

# 4. Start local stack (Docker must be running)
supabase start
supabase functions serve   # in a separate terminal
```

## Syncing From Dashboard

Pull the latest schema and function code from the Supabase dashboard:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync-supabase.ps1 -SyncSecretsTemplate
```

The script will warn if any deployed function is missing configuration in `config.toml`. Review `git diff supabase/` before committing.

## Running Tests

```bash
# Unit tests — no Docker required
cd supabase && deno test tests/unit/ --allow-read

# Integration tests — requires supabase start + supabase functions serve
cd supabase && deno test tests/integration/ --allow-net --allow-env
```
