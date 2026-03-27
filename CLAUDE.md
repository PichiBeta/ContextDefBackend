# ContextDefBackend

Backend for a language-learning reading platform. Supabase project (database + edge functions) is the primary concern. A small FastAPI service (`app/`) handles text chunking.

## Rules

- **Code-first**: All schema changes go through `supabase/migrations/`. Use `supabase migration new <name>` to create migration files. Never make schema changes directly in the dashboard without immediately pulling them.
- Do **NOT** rename `defintion-translation/` — the typo is the deployed URL slug.
- Do **NOT** rename `calculate_user_embedding/` — snake_case is the deployed URL slug. New functions use kebab-case.
- **When completing a task that changes schema or function logic**: write a decision log to `docs/agent-decisions/YYYYMMDD-<slug>.md` and suggest a commit message.

## Docs

- **Before modifying the readings pipeline or database schema**: see `docs/architecture/overview.md`.
- **Before modifying edge functions**: read `docs/edge-functions/overview.md` and the relevant function doc in `docs/edge-functions/`.
- **When design decisions are made** that establish or change patterns: update the relevant doc in `docs/`.
- **For onboarding/setup**: see `docs/onboarding.md`.

## Known Issues / Tech Debt

| Issue | Location | Notes |
|---|---|---|
| Hardcoded production URL in trigger | `migrations/*.sql` → `readings_enqueue_difficulty` | `pg_net` triggers require a hardcoded URL. Local dev does NOT fire this trigger against the local function. |
| Orphaned dashboard secrets | `supabase/.env.example` | `GEMINI_API_KEY` and `OPEN_API_KEY` exist in the dashboard but no function code references them. Investigate before removing. |
| Vault secrets not in migrations | Dashboard > Vault | `READINGS_DIFFICULTY_WEBHOOK_SECRET` must be set manually in the Vault — it is not seeded by any migration. |
