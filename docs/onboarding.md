# Engineer Onboarding

## Prerequisites

Install the following before you start:

| Tool | Why | How to install |
|---|---|---|
| [Supabase CLI](https://supabase.com/docs/guides/cli/getting-started) | DB migrations, local stack, function deploy | `scoop install supabase` (Windows) or `brew install supabase/tap/supabase` |
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Local Supabase stack (`supabase start`) | Download from docker.com — must be **running** before `supabase start` |
| [Deno 2.x](https://docs.deno.com/runtime/getting_started/installation/) | Run/type-check edge functions and tests | `scoop install deno` (Windows) or `brew install deno` |

Verify your installs:

```bash
supabase --version   # ≥ 2.x
deno --version       # ≥ 2.x
docker --version     # any recent version
```

## Step 1: Authenticate and Link the Project

```bash
supabase login
supabase link --project-ref irspwhgeyrojqluzgciu
```

`supabase link` connects your local CLI to the production project so that `db pull`, `functions deploy`, etc. target the right project by default.

## Step 2: Set Up Secrets

```bash
cp supabase/.env.example supabase/.env.local
```

Open `supabase/.env.local` and fill in the real values. Ask a teammate for the secrets — Supabase only stores digests, not plaintext. `supabase/.env.local` is gitignored and never committed.

The edge functions read this file automatically when you run `supabase functions serve`.

## Step 3: Start the Local Stack

**First**, capture any schema drift from the dashboard (this connects directly to the remote project — Docker not required):

```bash
supabase db pull remote_schema --project-ref irspwhgeyrojqluzgciu
```

If no schema drift exists, the command creates an empty migration file — it's safe to delete it. If it contains SQL, commit it before proceeding.

**Then**, make sure Docker Desktop is **open and running** and start the local stack:

```bash
supabase start
```

The first run pulls Docker images (~2 min). Subsequent starts are fast. When it's done, you'll see local URLs:

```
API URL:     http://127.0.0.1:54321
DB URL:      postgresql://postgres:postgres@127.0.0.1:54322/postgres
Studio URL:  http://127.0.0.1:54323
```

Open Studio at `http://127.0.0.1:54323` to browse the local DB.

## Step 4: Serve Edge Functions Locally

In a second terminal:

```bash
supabase functions serve
```

Functions are available at `http://127.0.0.1:54321/functions/v1/<function-name>`. They reload automatically when you edit a file.

## Step 5: Run Tests

```bash
# Unit tests — no Docker required
cd supabase && deno test tests/unit/ --allow-read

# Integration tests — requires supabase start + supabase functions serve
cd supabase && deno test tests/integration/ --allow-net --allow-env
```

All unit tests should pass. Integration tests require a running local stack and valid secrets in `supabase/.env.local`.

---

## Understanding the Architecture

- [CLAUDE.md](../CLAUDE.md) — rules, doc references, and known issues
- [docs/architecture/overview.md](architecture/overview.md) — data flow, database schema, triggers
- [docs/edge-functions/overview.md](edge-functions/overview.md) — conventions for edge function development

---

## Common Workflows

### Making a Schema Change

1. Make the change in local Studio or write SQL directly
2. `supabase db diff --schema public` — preview what changed
3. `supabase migration new <descriptive_name>` — create the migration file
4. `supabase db push --local` — apply new migration to the already-running local DB
5. Commit the migration file — **never skip this step**
6. When ready: `supabase db push` — apply to production (targets linked remote by default)

### Deploying a Function Change

Edit the function code in `supabase/functions/<name>/`, then:

```bash
supabase functions deploy <function-name>
```

Deploy only the changed function. Avoid `--all` until you've reviewed all diffs.

### Syncing Dashboard Changes to the Repo

If someone makes changes in the Supabase dashboard (avoid this, but it happens):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync-supabase.ps1 -SyncSecretsTemplate
```

Review `git diff supabase/` and commit the changes.

### Adding a New Edge Function

See the full checklist in [docs/edge-functions/overview.md](edge-functions/overview.md#creating-a-new-function). Deploy when ready:

```bash
supabase functions deploy <function-name>
```

---

## Common Pitfalls

| Problem | Cause | Fix |
|---|---|---|
| `supabase start` fails | Docker Desktop not running | Start Docker Desktop, wait for it to be ready, retry |
| Function returns 500 locally | Missing secret in `.env.local` | Check `supabase/.env.local` has the right keys for that function |
| DB trigger doesn't fire locally | The production webhook URL is hardcoded | Call `process-reading` directly in integration tests instead |
| `deno test` can't resolve imports | Running from wrong directory | Run `cd supabase && deno test ...` not from repo root |
| Function missing from `supabase functions serve` | No `deno.json` in function directory | Add a `deno.json` (even an empty `{}`) to the function directory |
| Schema drift after dashboard changes | Someone edited the schema in the dashboard | Run the sync script and commit the new migration |

---

## Storage Buckets

The `readings` storage bucket is created by migration (`20260326000000_create_readings_bucket.sql`) and configured in `config.toml` for local dev. No manual setup needed.
