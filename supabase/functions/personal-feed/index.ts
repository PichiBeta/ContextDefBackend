import { requireUser } from "../_shared/auth.ts";

// TODO: tighten in production (echo allowed origin + add Vary: Origin)
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Vary": "Origin",
};

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}

Deno.serve(async (req) => {
  // --- CORS preflight ---
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  try {
    if (req.method !== "GET") {
      return jsonResponse({ ok: false, error: "Method not allowed" }, 405);
    }

    // --- Require & verify caller JWT ---
    const auth = await requireUser(req);
    if (auth instanceof Response) return auth;
    const { user, supabase } = auth;

    // --- Parse query params ---
    const url = new URL(req.url);
    const rawLimit = parseInt(url.searchParams.get("limit") ?? "20", 10);
    const rawOffset = parseInt(url.searchParams.get("offset") ?? "0", 10);
    const limit = Math.min(Math.max(isNaN(rawLimit) ? 20 : rawLimit, 1), 100);
    const offset = isNaN(rawOffset) || rawOffset < 0 ? 0 : rawOffset;

    // --- Fetch user profile to check for an existing embedding ---
    const { data: profile, error: profileErr } = await supabase
      .from("profiles")
      .select("embedding, num_vectors")
      .eq("id", user.id)
      .single();

    if (profileErr) {
      console.error("Failed to fetch profile:", profileErr);
      return jsonResponse(
        { ok: false, error: "Failed to fetch profile", detail: profileErr.message },
        500,
      );
    }

    const hasEmbedding = profile && (profile.num_vectors ?? 0) > 0 && profile.embedding != null;
    const feedType = hasEmbedding ? "personalized" : "discovery";

    // --- Call the RPC ---
    const { data: feed, error: rpcErr } = await supabase.rpc("get_personal_feed", {
      p_user_embedding: hasEmbedding ? profile.embedding : null,
      p_limit: limit,
      p_offset: offset,
    });

    if (rpcErr) {
      console.error("get_personal_feed RPC failed:", rpcErr);
      return jsonResponse(
        { ok: false, error: "Failed to fetch feed", detail: rpcErr.message },
        500,
      );
    }

    return jsonResponse({ ok: true, feed_type: feedType, feed: feed ?? [] });
  } catch (err) {
    console.error("Unhandled error:", err);
    return jsonResponse(
      { ok: false, error: "Internal error", detail: String((err as Error)?.message ?? err) },
      500,
    );
  }
});
