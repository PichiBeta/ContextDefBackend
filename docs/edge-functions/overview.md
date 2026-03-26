# Edge Functions — Overview

## Imports: Bare Specifiers

Every function directory has a `deno.json` with an `imports` map. Code files use **bare specifier aliases** — never inline `npm:` or `jsr:` URLs.

deno.json:
```json
{ "imports": { "supabase": "jsr:@supabase/supabase-js@2" } }
```

```ts
// index.ts — correct
import { createClient } from "supabase";

// index.ts — wrong (inline specifier)
import { createClient } from "jsr:@supabase/supabase-js@2";
```

Only `deno.json` contains the versioned specifier. The bare specifier key should be a short, descriptive name (e.g., `"supabase"`, `"anthropic"`, `"wink-tokenizer"`).

**`@supabase/functions-js`** is only needed in functions that use `import "@supabase/functions-js/edge-runtime.d.ts"` for edge runtime type declarations. Do not add it to every function. Currently used by: `calculate-difficulty`, `ocr-extract`.

## Auth: Custom Auth (verify_jwt = false)

All functions set `verify_jwt = false` in `config.toml`. The project uses a **publishable key** (not anon key) on the client side — Supabase's legacy JWT verification is OFF. Auth is handled in function code instead.

### Two auth patterns

**1. Client-facing APIs** — Require `Authorization: Bearer <token>` header, verify the user exists via `supabase.auth.getUser()`.

Used by: `create-reading`, `defintion-translation`

```ts
const jwt = parseBearer(req);
if (!jwt) return jsonResponse({ error: "Missing bearer token" }, 401);

const supabase = createClient(SUPABASE_URL, SUPABASE_PUBLIC_KEY, {
  global: { headers: { Authorization: `Bearer ${jwt}`, apikey: SUPABASE_PUBLIC_KEY } },
});

const { data: userData, error: userErr } = await supabase.auth.getUser();
if (userErr || !userData?.user) return jsonResponse({ error: "Unauthorized" }, 401);
```

**2. DB-triggered webhooks** — Require `x-webhook-secret` header matching a Vault secret.

Used by: `process-reading`, `calculate-difficulty`

```ts
const secret = req.headers.get("x-webhook-secret");
if (!secret || secret !== WEBHOOK_SECRET) {
  return new Response(JSON.stringify({ ok: false, error: "Unauthorized" }), { status: 401 });
}
```

### Functions without auth (known gap)

`ocr-extract` and `calculate_user_embedding` currently have no auth checks. This is flagged as tech debt.

## Edge Functions Inventory

| Function | Type | Auth | Key Secrets |
|---|---|---|---|
| `process-reading` | Webhook (DB trigger) | `x-webhook-secret` header | `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `READINGS_DIFFICULTY_WEBHOOK_SECRET`, `HF_API_KEY` |
| `create-reading` | API (client) | Bearer token → `auth.getUser()` | `SUPABASE_URL`, `SUPABASE_ANON_KEY` |
| `calculate-difficulty` | Webhook (DB trigger) | `x-webhook-secret` header | `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `READINGS_DIFFICULTY_WEBHOOK_SECRET` |
| `ocr-extract` | API (client) | None (known gap) | `AZURE_DOC_INTEL_ENDPOINT`, `AZURE_DOC_INTEL_KEY` |
| `defintion-translation` | API (client) | Bearer token → `auth.getUser()` | `ANTHROPIC_API_KEY` |
| `calculate_user_embedding` | Internal API | None (known gap) | `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` |

## Function Naming

- New functions: **kebab-case** (e.g., `my-new-function/`)
- Exception: `calculate_user_embedding` — snake_case because it was dashboard-created. Renaming changes the deployed URL slug, so keep as-is.

## Workspace Config

`supabase/deno.json` defines a Deno 2 workspace with all function directories as members. This allows test files (which live outside function directories) to resolve imports across function boundaries.

- Tests must run from the `supabase/` directory: `cd supabase && deno test ...`
- The workspace-level imports (`@std/assert`, `supabase`) are for test files and shared utilities, not for function code.
- Each function's own `deno.json` governs its imports independently.

## Creating a New Function

1. `supabase functions new <name>` — scaffolds `index.ts`, `deno.json`, and auto-appends `[functions.<name>]` to `config.toml`
2. Set `verify_jwt = false` in `config.toml` (unless you have a specific reason for legacy JWT)
3. Add the function to the `workspace` array in `supabase/deno.json`
4. Implement one of the two auth patterns above in your function code
5. Create a doc file: `docs/edge-functions/<name>.md`
