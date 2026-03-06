import Anthropic from "npm:@anthropic-ai/sdk@0.39.0";
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type"
};
Deno.serve(async (req)=>{
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response("ok", {
      headers: corsHeaders
    });
  }
  try {
    const { selection, context, language } = await req.json();
    if (!selection || !context || !language) {
      return new Response(JSON.stringify({
        error: "selection, context, and language are required."
      }), {
        status: 400,
        headers: {
          ...corsHeaders,
          "Content-Type": "application/json"
        }
      });
    }
    const anthropic = new Anthropic({
      apiKey: Deno.env.get("ANTHROPIC_API_KEY")
    });
    const prompt = `You are a concise dictionary and translation assistant.

The user has selected the word or phrase: "${selection}"
It appears in this context: "${context}"

Your task:
1. Detect the language of the selection and context.
2. Write a definition of "${selection}" in that same language using exactly 2–3 bullet points. Each bullet should be one sentence, clear and direct.
3. Provide a brief translation of "${selection}" into ${language}.

Respond ONLY with raw JSON (no markdown, no code fences, no backticks, no extra keys):
{
  "definition": ["bullet 1", "bullet 2", "bullet 3"],
  "translation": "translation here"
}`;
    const message = await anthropic.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 512,
      messages: [
        {
          role: "user",
          content: prompt
        }
      ]
    });
    // Extract the text content from the response
    const rawText = message.content.filter((block)=>block.type === "text").map((block)=>block.text).join("");
    // Parse the JSON Claude returns
    const cleaned = rawText.replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/i, "").trim();
    const parsed = JSON.parse(cleaned);
    return new Response(JSON.stringify(parsed), {
      status: 200,
      headers: {
        ...corsHeaders,
        "Content-Type": "application/json"
      }
    });
  } catch (err) {
    console.error("Error in define-and-translate function:", err);
    return new Response(JSON.stringify({
      error: "Internal server error",
      detail: String(err)
    }), {
      status: 500,
      headers: {
        ...corsHeaders,
        "Content-Type": "application/json"
      }
    });
  }
});
