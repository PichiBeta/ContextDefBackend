import "@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "jsr:@supabase/supabase-js@2";

/*
v1.0 (updated) - Classic readability: single scalar difficulty score derived
from Flesch–Kincaid Grade Level. Works only for English text. Normalized 0-100

Ideal version: more nuanced difficulty score, compatible with all languages the app supports
*/

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const WEBHOOK_SECRET = Deno.env.get("READINGS_DIFFICULTY_WEBHOOK_SECRET")!;

const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY);

function countSyllables(word: string): number {
  const w = word
    .toLowerCase()
    .replace(/[^a-z]/g, "")
    .trim();

  if (!w) return 0;
  if (w.length <= 3) return 1;

  const noSilentE = w.replace(/e\b/, "");
  const vowelGroups = noSilentE.match(/[aeiouy]+/g);
  let count = vowelGroups ? vowelGroups.length : 1;

  if (/(ia|io|eo)$/.test(w)) count += 1;
  if (/(tion|sion)$/.test(w)) count -= 1;

  return Math.max(1, count);
}

function tokenizeSentences(text: string): string[] {
  const sentences = text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);

  return sentences.length ? sentences : [text.trim()].filter(Boolean);
}

function tokenizeWords(text: string): string[] {
  return text
    .toLowerCase()
    .match(/[a-z]+(?:'[a-z]+)?/g) ?? [];
}

function computeDifficultyScore(text: string): number {
  const sentences = tokenizeSentences(text);
  const words = tokenizeWords(text);

  const sentenceCount = sentences.length || 1;
  const wordCount = words.length || 1;

  let syllableCount = 0;
  for (const w of words) syllableCount += countSyllables(w);

  const wordsPerSentence = wordCount / sentenceCount;
  const syllablesPerWord = syllableCount / wordCount;

  // Flesch–Kincaid Grade Level
  const fleschKincaidGrade =
    0.39 * wordsPerSentence + 11.8 * syllablesPerWord - 15.59;

  // Normalize: 0 = easy, 100 = hard (cap at grade 30)
  return Math.min(100, Math.max(0, (fleschKincaidGrade / 30) * 100));
}

Deno.serve(async (req) => {
  try {
    // Authenticate caller is DB trigger
    const secret = req.headers.get("x-webhook-secret");
    if (!secret || secret !== WEBHOOK_SECRET) {
      return new Response(
        JSON.stringify({ ok: false, error: "Unauthorized" }),
        { status: 401, headers: { "Content-Type": "application/json" } },
      );
    }

    const { reading_id, storage_path, content_updated_at } = await req.json();

    if (typeof reading_id !== "string" || reading_id.trim().length === 0) {
      throw new Error('Invalid payload: expected non-empty "reading_id"');
    }
    if (typeof storage_path !== "string" || storage_path.trim().length === 0) {
      throw new Error('Invalid payload: expected non-empty "storage_path"');
    }
    if (typeof content_updated_at !== "string" || content_updated_at.trim().length === 0) {
      throw new Error('Invalid payload: expected non-empty "content_updated_at"');
    }

    // Fetch content from Storage (S3)
    const { data: file, error: dlErr } = await supabase.storage
      .from("readings")
      .download(storage_path);

    if (dlErr || !file) throw dlErr ?? new Error("Failed to download reading content");

    const content = await file.text();
    if (!content || content.trim().length === 0) {
      throw new Error("Downloaded content is empty");
    }

    const difficulty = computeDifficultyScore(content);

    // Use content_updated_at to avoid writing results for stale content.
    const { data: updated, error: updErr } = await supabase
      .from("readings")
      .update({
        difficulty: Math.round(difficulty),
        status: "processed",
      })
      .eq("id", reading_id)
      .eq("content_updated_at", content_updated_at)
      .select("id")
      .maybeSingle();

    if (updErr) throw updErr;

    // If nothing matched, content was likely updated again; don't overwrite newer results.
    if (!updated) {
      return new Response(
        JSON.stringify({ ok: false, error: "Stale update", detail: "content_updated_at mismatch" }),
        { status: 409, headers: { "Content-Type": "application/json" } },
      );
    }

    return new Response(
      JSON.stringify({
        ok: true,
        difficulty, // 0 (easy) → 100 (hard)
      }),
      { headers: { "Content-Type": "application/json" } },
    );
  } catch (err) {
    // Best-effort failure update
    try {
      const body = await req.clone().json();
      if (body?.reading_id && typeof body.reading_id === "string") {
        await supabase
          .from("readings")
          .update({
            status: "failed",
          })
          .eq("id", body.reading_id);
      }
    } catch (_) {}

    return new Response(
      JSON.stringify({
        ok: false,
        detail: String((err as Error)?.message ?? err),
      }),
      { status: 500, headers: { "Content-Type": "application/json" } },
    );
  }
});
