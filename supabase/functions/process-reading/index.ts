import { createClient } from "supabase";
import { requireWebhookSecret } from "../_shared/auth.ts";
import { tokenizeReading } from "./tokenize-reading.ts";
import { calculateDifficulty } from "./calculate-difficulty.ts";
import { calculateReadingEmbedding } from "./calculate-reading-embedding.ts";
/*
Process reading (orchestrator):
- verify webhook secret
- download raw content once
- generate reading structure v1 and upload to Storage
- compute difficulty score (0–100)
- compute reading embedding from sentence chunks
- mark processed ONLY if all succeeded (single DB update)
*/
const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const WEBHOOK_SECRET = Deno.env.get("READINGS_DIFFICULTY_WEBHOOK_SECRET");

const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY);

Deno.serve(async (req) => {
  let reading_id: string | null = null;

  try {
    // 1) Verify caller (DB trigger via pg_net) using shared secret
    const authResult = requireWebhookSecret(req, WEBHOOK_SECRET);
    if (authResult instanceof Response) return authResult;

    // 2) Parse payload
    const body = await req.json();
    if (typeof body.reading_id !== "string" || body.reading_id.trim().length === 0) {
      throw new Error('Invalid payload: expected non-empty "reading_id"');
    }
    if (typeof body.storage_path !== "string" || body.storage_path.trim().length === 0) {
      throw new Error('Invalid payload: expected non-empty "storage_path"');
    }
    if (typeof body.language_code !== "string" || body.reading_id.trim().length === 0) {
      throw new Error('Invalid payload: expected non-empty "language_code"');
    }
    if (
      typeof body.content_updated_at !== "string" ||
      body.content_updated_at.trim().length === 0
    ) {
      throw new Error('Invalid payload: expected non-empty "content_updated_at"');
    }

    reading_id = body.reading_id.trim() as string;
    const language_code = body.language_code.trim();
    const storage_path = body.storage_path.trim();
    const content_updated_at = body.content_updated_at.trim();

    // 3) Download raw text from Storage
    const { data: file, error: dlErr } = await supabase.storage
      .from("readings")
      .download(storage_path);
    if (dlErr || !file) {
      throw dlErr ?? new Error("Failed to download reading content");
    }

    const text = await file.text();
    if (!text || text.trim().length === 0) {
      throw new Error("Downloaded content is empty");
    }

    const ctx = { language_code, text };

    // 4) Deterministic structure + difficulty (pure)
    const structure = tokenizeReading(ctx);
    const { score } = calculateDifficulty(ctx);

    // Inject identifiers (module stays pure; orchestrator adds identity)
    structure.reading_id = reading_id;
    structure.text = structure.text ?? {};
    structure.text.storage_path = storage_path;

    // 5) Upload structure JSON
    const structurePath = `${storage_path}.structure.v1.json`;
    const { error: upErr } = await supabase.storage.from("readings").upload(
      structurePath,
      new Blob([JSON.stringify(structure)], { type: "application/json" }),
      { upsert: true, contentType: "application/json" },
    );
    if (upErr) throw upErr;

    // 6) Compute reading embedding from current processing context
    const { embedding, chunks_processed } = await calculateReadingEmbedding({
      text,
      structure,
    });

    // STEP 7: Update the reading row with the embedding
    // Single DB update (guard against stale content)
    const difficultyInt = Math.round(score);
    const { data: updated, error: updErr } = await supabase
      .from("readings")
      .update({
        difficulty: difficultyInt,
        embedding,
        status: "processed",
        error_message: null,
      })
      .eq("id", reading_id)
      .eq("content_updated_at", content_updated_at)
      .select("id")
      .maybeSingle();

    if (updErr) throw updErr;
    if (!updated) {
      return new Response(
        JSON.stringify({
          ok: false,
          error: "Stale update",
          detail: "content_updated_at mismatch",
        }),
        {
          status: 409,
          headers: { "Content-Type": "application/json" },
        },
      );
    }

    return new Response(
      JSON.stringify({
        ok: true,
        difficulty: difficultyInt,
        structure_path: structurePath,
        chunks_processed,
      }),
      {
        headers: { "Content-Type": "application/json" },
      },
    );
  } catch (err) {
    // Best-effort failure update (fail-fast model)
    try {
      if (reading_id) {
        await supabase
          .from("readings")
          .update({
            status: "failed",
            error_message: String((err as Error)?.message ?? err),
          })
          .eq("id", reading_id);
      }
    } catch (_) {
      // no-op
    }

    return new Response(
      JSON.stringify({ ok: false, detail: String((err as Error)?.message ?? err) }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
});

