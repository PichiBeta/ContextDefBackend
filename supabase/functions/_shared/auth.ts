import { createClient, type SupabaseClient, type User } from "supabase";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SUPABASE_PUBLIC_KEY =
  Deno.env.get("SUPABASE_PUBLISHABLE_KEY") ??
  Deno.env.get("SUPABASE_ANON_KEY") ??
  "";

function unauthorized(message = "Unauthorized"): Response {
  return new Response(JSON.stringify({ ok: false, error: message }), {
    status: 401,
    headers: { "Content-Type": "application/json" },
  });
}

/**
 * Validates the Bearer JWT in the Authorization header.
 * Returns { user, supabase } on success — the client is scoped to the caller's JWT so RLS applies.
 * Returns a 401 Response on failure — callers should return it immediately:
 *   const auth = await requireUser(req);
 *   if (auth instanceof Response) return auth;
 */
export async function requireUser(
  req: Request,
): Promise<{ user: User; supabase: SupabaseClient } | Response> {
  const authHeader = req.headers.get("authorization") ?? "";
  if (!authHeader.startsWith("Bearer ")) return unauthorized("Missing bearer token");
  const token = authHeader.slice(7).trim();
  if (!token) return unauthorized("Missing bearer token");

  const supabase = createClient(SUPABASE_URL, SUPABASE_PUBLIC_KEY, {
    global: {
      headers: {
        Authorization: `Bearer ${token}`,
        apikey: SUPABASE_PUBLIC_KEY,
      },
    },
  });

  const { data, error } = await supabase.auth.getUser();
  if (error || !data?.user) {
    console.error("auth.getUser failed:", error);
    return unauthorized();
  }

  return { user: data.user, supabase };
}

/**
 * Validates the x-webhook-secret header against the expected secret.
 * Returns true on success.
 * Returns a 401 Response on failure — callers should return it immediately:
 *   const authResult = requireWebhookSecret(req, WEBHOOK_SECRET);
 *   if (authResult instanceof Response) return authResult;
 */
export function requireWebhookSecret(
  req: Request,
  expectedSecret: string | undefined,
): true | Response {
  const secret = req.headers.get("x-webhook-secret");
  if (!secret || secret !== expectedSecret) return unauthorized();
  return true;
}
