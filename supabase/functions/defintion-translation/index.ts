import Anthropic from "anthropic";
import { getPrompt } from "./prompts.ts";
import { fetchWiktionary } from "./wiktionary.ts";
import { requireUser } from "../_shared/auth.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

interface RequestBody {
  selection: string;        // The word or phrase to define
  context: string;          // The sentence/paragraph it comes from
  language: string;         // Target language for the translation (e.g. "English", "Spanish")
  language_code: string;    // Source language of the reading
  reading_id: string;
  selection_start: number;
  selection_end: number;
}

interface LLMResult {
  definition: string[];  // 2–3 bullet points in the source language
  translation: string;   // Brief translation in the target language
}

interface DefinitionResponse extends LLMResult {
  part_of_speech: string | null;
  examples: Array<{ text: string; translation?: string }>;
}

async function callLLM(
  anthropic: Anthropic,
  prompt: string,
): Promise<LLMResult> {
  const message = await anthropic.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 512,
    messages: [{ role: "user", content: prompt }],
  });

  const rawText = message.content
    .filter((block) => block.type === "text")
    .map((block) => (block as { type: "text"; text: string }).text)
    .join("");

  const cleaned = rawText
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```\s*$/i, "")
    .trim();

  return JSON.parse(cleaned) as LLMResult;
}

Deno.serve(async (req: Request) => {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const auth = await requireUser(req);
    if (auth instanceof Response) return auth;
    const { supabase } = auth;

    const {
      selection,
      context,
      language,
      language_code,
      reading_id,
      selection_start,
      selection_end,
    } = (await req.json()) as RequestBody;

    if (
      !selection ||
      !context ||
      !language ||
      !language_code ||
      !reading_id ||
      selection_start === undefined ||
      selection_end === undefined
    ) {
      return new Response(
        JSON.stringify({
          error:
            "selection, context, language, language_code, reading_id, selection_start, and selection_end are required.",
        }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    // Try cache hit first: update last_accessed and return the row in one statement.
    const { data: cachedRow, error: cacheError } = await supabase
      .from("words_lookup_cache")
      .update({ last_accessed: new Date().toISOString() })
      .eq("reading_id", reading_id)
      .eq("selection_start", selection_start)
      .eq("selection_end", selection_end)
      .select("definition, translation, part_of_speech, examples")
      .maybeSingle();

    if (cacheError) {
      throw cacheError;
    }

    if (cachedRow) {
      return new Response(
        JSON.stringify({
          definition: cachedRow.definition,
          translation: cachedRow.translation,
          part_of_speech: cachedRow.part_of_speech ?? null,
          examples: JSON.parse(cachedRow.examples ?? "[]"),
        }),
        {
          status: 200,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Cache miss — fetch LLM and Wiktionary in parallel.
    const anthropic = new Anthropic({
      apiKey: Deno.env.get("ANTHROPIC_API_KEY")!,
    });

    const prompt = getPrompt(language_code, selection, context, language);

    const [llmSettled, wiktSettled] = await Promise.allSettled([
      callLLM(anthropic, prompt),
      fetchWiktionary(selection, language_code),
    ]);

    // LLM is the critical path — if it failed, throw
    if (llmSettled.status === "rejected") {
      throw llmSettled.reason;
    }

    const parsed: LLMResult = llmSettled.value;
    const wikt = wiktSettled.status === "fulfilled" ? wiktSettled.value : null;

    const response: DefinitionResponse = {
      definition: parsed.definition,
      translation: parsed.translation,
      part_of_speech: wikt?.partOfSpeech ?? null,
      examples: wikt?.examples ?? [],
    };

    const { error: insertError } = await supabase
      .from("words_lookup_cache")
      .insert({
        selection,
        definition: parsed.definition,
        translation: parsed.translation,
        part_of_speech: response.part_of_speech,
        examples: JSON.stringify(response.examples),
        reading_id,
        selection_start,
        selection_end,
        native_language: language_code,
      });

    if (insertError) {
      // Handle rare race condition where another request inserted the row first
      const msg = String(insertError.message ?? "");
      const details = String((insertError as { details?: string }).details ?? "");

      const isUniqueViolation =
        msg.includes("duplicate key") ||
        msg.includes("unique") ||
        details.includes("already exists");

      if (!isUniqueViolation) {
        throw insertError;
      }
    }

    return new Response(JSON.stringify(response), {
      status: 200,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (err) {
    console.error("Error in define-and-translate function:", err);

    return new Response(
      JSON.stringify({
        error: "Internal server error",
        detail: String(err),
      }),
      {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      },
    );
  }
});
