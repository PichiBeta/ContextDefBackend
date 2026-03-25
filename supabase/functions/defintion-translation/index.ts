import Anthropic from "anthropic";
import { createClient } from "supabase";
import { getPrompt } from "./prompts.ts";

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

interface DefinitionResponse {
  definition: string[];  // 2–3 bullet points in the source language
  translation: string;   // Brief translation in the target language
}

Deno.serve(async (req: Request) => {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const authHeader = req.headers.get("Authorization");

    if (!authHeader) {
      return new Response(
        JSON.stringify({ error: "Authentication required." }),
        { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    const supabase = createClient(
      Deno.env.get("SUPABASE_URL")!,
      Deno.env.get("SUPABASE_ANON_KEY")!,
      {
        global: {
          headers: { Authorization: authHeader },
        },
      },
    );

    // Validate JWT
    const { data: userData, error: authError } = await supabase.auth.getUser();

    if (authError || !userData?.user) {
      return new Response(
        JSON.stringify({ error: "Invalid authentication token." }),
        { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

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
      .select("definition, translation")
      .maybeSingle();

    if (cacheError) {
      throw cacheError;
    }

    if (cachedRow) {
      return new Response(
        JSON.stringify({
          definition: cachedRow.definition,
          translation: cachedRow.translation,
        }),
        {
          status: 200,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    const anthropic = new Anthropic({
      apiKey: Deno.env.get("ANTHROPIC_API_KEY")!,
    });

    const prompt = getPrompt(language_code, selection, context, language);

    const message = await anthropic.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 512,
      messages: [{ role: "user", content: prompt }],
    });

    // Extract the text content from the response
    const rawText = message.content
      .filter((block) => block.type === "text")
      .map((block) => (block as { type: "text"; text: string }).text)
      .join("");

    // Parse the JSON Claude returns
    const cleaned = rawText
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```\s*$/i, "")
      .trim();

    const parsed: DefinitionResponse = JSON.parse(cleaned);

    const { error: insertError } = await supabase
      .from("words_lookup_cache")
      .insert({
        selection,
        definition: parsed.definition,
        translation: parsed.translation,
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

    return new Response(JSON.stringify(parsed), {
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