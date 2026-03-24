import { createClient } from "jsr:@supabase/supabase-js@2";


 // We explicitly require a Bearer access token and verify it via auth.getUser().
 // All DB + Storage writes run with the caller's JWT so RLS policies apply.

type Visibility = "private" | "unlisted" | "public";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
// Use whichever env var you actually set in Dashboard -> Edge Functions -> Secrets.
// Prefer publishable key if that's your project convention.
const SUPABASE_PUBLIC_KEY =
  Deno.env.get("SUPABASE_PUBLISHABLE_KEY") ??
  Deno.env.get("SUPABASE_ANON_KEY") ??
  "";

if (!SUPABASE_URL || !SUPABASE_PUBLIC_KEY) {
  // Fail fast with a clear log if secrets are misconfigured.
  console.error("Missing SUPABASE_URL or SUPABASE_PUBLIC_KEY (publishable/anon) in env");
  throw new Error("Server misconfigured");
}

// TODO: tighten in production (echo allowed origin + add Vary: Origin)
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Vary": "Origin",
};

function jsonResponse(body: unknown, status = 200, extraHeaders: HeadersInit = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      ...corsHeaders,
      "Content-Type": "application/json",
      ...extraHeaders,
    },
  });
}

function parseBearer(req: Request): string | null {
  const authHeader = req.headers.get("authorization") ?? "";
  if (!authHeader.startsWith("Bearer ")) return null;
  const token = authHeader.slice(7).trim();
  return token.length ? token : null;
}

Deno.serve(async (req) => {
  // --- CORS preflight ---
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  try {
    if (req.method !== "POST") {
      return jsonResponse({ ok: false, error: "Method not allowed" }, 405);
    }

    // --- Require & verify caller JWT (works when legacy verify is OFF) ---
    const jwt = parseBearer(req);
    if (!jwt) {
      return jsonResponse({ ok: false, error: "Missing bearer token" }, 401);
    }

    // Use caller token so RLS applies. Include apikey as well (recommended).
    const supabase = createClient(SUPABASE_URL, SUPABASE_PUBLIC_KEY, {
      global: {
        headers: {
          Authorization: `Bearer ${jwt}`,
          apikey: SUPABASE_PUBLIC_KEY,
        },
      },
    });

    const { data: userData, error: userErr } = await supabase.auth.getUser();
    if (userErr || !userData?.user) {
      console.error("auth.getUser failed:", userErr);
      return jsonResponse({ ok: false, error: "Unauthorized" }, 401);
    }

    const owner_id = userData.user.id;

    // --- Parse payload safely ---
    let payload: unknown;
    try {
      payload = await req.json();
    } catch {
      return jsonResponse({ ok: false, error: "Invalid JSON body" }, 400);
    }

    const { title, genre, language_code, visibility, content } = payload as Record<
      string,
      unknown
    >;

    if (typeof title !== "string" || title.trim().length === 0) {
      return jsonResponse({ ok: false, error: 'Invalid payload: non-empty "title" required' }, 400);
    }
    if (typeof genre !== "string" || genre.trim().length === 0) {
      return jsonResponse({ ok: false, error: 'Invalid payload: non-empty "genre" required' }, 400);
    }
    if (typeof language_code !== "string" || language_code.trim().length === 0) {
      return jsonResponse(
        { ok: false, error: 'Invalid payload: non-empty "language_code" required' },
        400,
      );
    }
    if (
      typeof visibility !== "string" ||
      !["private", "unlisted", "public"].includes(visibility)
    ) {
      return jsonResponse(
        {
          ok: false,
          error: 'Invalid payload: visibility must be "private" | "unlisted" | "public"',
        },
        400,
      );
    }
    if (typeof content !== "string" || content.trim().length === 0) {
      return jsonResponse({ ok: false, error: 'Invalid payload: non-empty "content" required' }, 400);
    }

    const v = visibility as Visibility;
    const content_preview = content.slice(0, 70);

    // --- 1) Create DB row ---
    // Assumes your table defaults handle: status, is_deleted, deleted_at, error_message, content_updated_at, etc.
    const { data: reading, error: insertErr } = await supabase
      .from("readings")
      .insert({
        owner_id,
        title: title.trim(),
        genre: genre.trim(),
        language_code: language_code.trim(),
        visibility: v,
        content_preview,
      })
      .select("id")
      .single();

    if (insertErr || !reading?.id) {
      console.error("Insert failed:", insertErr);
      return jsonResponse(
        { ok: false, error: "Insert failed", detail: insertErr?.message ?? "Unknown error" },
        500,
      );
    }

    const reading_id = String(reading.id);

    // --- 1.5) Add to user's Library (explicit membership) ---
    const { error: saveErr } = await supabase
      .from("user_saved_readings")
      .insert({
        user_id: owner_id,
        reading_id,
      });

    if (saveErr) {
      console.error("Failed to create library entry:", saveErr);

      // Best-effort: clean up reading to avoid orphaned row
      await supabase.from("readings").delete().eq("id", reading_id);

      return jsonResponse(
        { ok: false, error: "Failed to create library entry", detail: saveErr.message },
        500,
      );
    }

    // --- 2) Upload full content to Storage ---
    const objectName = `${reading_id}`;
    const bytes = new TextEncoder().encode(content);

    const { error: uploadErr } = await supabase.storage
      .from("readings")
      .upload(objectName, bytes, {
        contentType: "text/plain; charset=utf-8",
        upsert: false,
      });

    if (uploadErr) {
      console.error("Upload failed:", uploadErr);

      // Best-effort: mark failed so UI isn't stuck
      const { error: markErr } = await supabase
        .from("readings")
        .update({
          status: "failed",
          error_message: `Upload failed: ${uploadErr.message}`,
        })
        .eq("id", reading_id);

      if (markErr) console.error("Failed to mark reading as failed:", markErr);

      return jsonResponse(
        { ok: false, reading_id, error: "Upload failed", detail: uploadErr.message },
        500,
      );
    }

    // --- 3) Mark uploaded (trigger can process) ---
    const { error: updErr } = await supabase
      .from("readings")
      .update({
        status: "uploaded",
        content_updated_at: new Date().toISOString(),
        error_message: null,
      })
      .eq("id", reading_id);

    if (updErr) {
      console.error("Update failed:", updErr);
      return jsonResponse(
        { ok: false, reading_id, error: "Update failed", detail: updErr.message },
        500,
      );
    }

    return jsonResponse({ ok: true, reading_id, status: "uploaded" }, 200);
  } catch (err) {
    console.error("Unhandled error:", err);
    return jsonResponse(
      { ok: false, error: "Internal error", detail: String((err as Error)?.message ?? err) },
      500,
    );
  }
});
