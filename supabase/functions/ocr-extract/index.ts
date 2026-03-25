import "@supabase/functions-js/edge-runtime.d.ts";

const corsHeaders: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, apikey, content-type, x-client-info, x-supabase-client-platform",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

function json(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders },
  });
}

const AZURE_ENDPOINT = Deno.env.get("AZURE_DOC_INTEL_ENDPOINT")!;
const AZURE_KEY = Deno.env.get("AZURE_DOC_INTEL_KEY")!;

const AZURE_ANALYZE_URL =
  `${AZURE_ENDPOINT}/documentintelligence/documentModels/prebuilt-read:analyze?api-version=2024-02-29-preview`;

const MAX_FILES = 10;
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const MAX_PDF_BYTES = 20 * 1024 * 1024;
const MAX_TOTAL_BYTES = 80 * 1024 * 1024;

// Safe default for Azure F0 (2 requests/sec). Bump later on S0.
const OCR_CONCURRENCY = 2;

async function pollAzure(operationUrl: string) {
  while (true) {
    const resp = await fetch(operationUrl, {
      headers: {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
      },
    });

    const data = await resp.json();

    if (data.status === "succeeded") return data;

    if (data.status === "failed") {
      throw new Error("Azure OCR failed");
    }

    await new Promise((r) => setTimeout(r, 1000));
  }
}

function normalizeParagraphs(paragraphs: { content: string }[]) {
  return paragraphs
    .map((p) => p.content.replace(/\s+/g, " ").trim())
    .filter((p) => p.length > 0)
    .join("\n\n");
}

// Concurrency-limited mapper that preserves order
async function mapWithConcurrency<T, R>(
  items: T[],
  concurrency: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let nextIndex = 0;

  async function worker() {
    while (true) {
      const i = nextIndex++;
      if (i >= items.length) return;
      results[i] = await fn(items[i], i);
    }
  }

  const workers = Array.from(
    { length: Math.min(concurrency, items.length) },
    () => worker(),
  );

  await Promise.all(workers);
  return results;
}

// Per-file OCR extracted so we can run it in parallel
async function ocrSingleFile(f: File): Promise<string> {
  const isPDF = f.type === "application/pdf";

  const analyzeUrl = isPDF
    ? `${AZURE_ANALYZE_URL}&pages=1-10`
    : AZURE_ANALYZE_URL;

  const submitResp = await fetch(analyzeUrl, {
    method: "POST",
    headers: {
      "Ocp-Apim-Subscription-Key": AZURE_KEY,
      "Content-Type": f.type,
    },
    body: await f.arrayBuffer(),
  });

  if (submitResp.status !== 202) {
    const err = await submitResp.text();
    throw new Error(`Azure OCR submit failed: ${err}`);
  }

  const operationUrl = submitResp.headers.get("operation-location");
  if (!operationUrl) {
    throw new Error("Missing Azure operation location");
  }

  const result = await pollAzure(operationUrl);
  const paragraphs = result.analyzeResult?.paragraphs ?? [];
  return normalizeParagraphs(paragraphs);
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  if (req.method !== "POST") {
    return json(405, { error: "Method not allowed" });
  }

  const contentType = req.headers.get("content-type") ?? "";
  if (!contentType.includes("multipart/form-data")) {
    return json(415, { error: "Expected multipart/form-data" });
  }

  const form = await req.formData();
  const files = form.getAll("files");

  if (files.length === 0) {
    return json(400, { error: "No files uploaded" });
  }

  if (files.length > MAX_FILES) {
    return json(400, { error: "Too many files (max 10)" });
  }

  let totalBytes = 0;

  for (const f of files) {
    if (!(f instanceof File)) {
      return json(400, { error: "Invalid file upload" });
    }

    totalBytes += f.size;

    const isPDF = f.type === "application/pdf";

    if (isPDF && f.size > MAX_PDF_BYTES) {
      return json(400, {
        error: `PDF ${f.name} exceeds 20MB limit`,
      });
    }

    if (!isPDF && f.size > MAX_IMAGE_BYTES) {
      return json(400, {
        error: `Image ${f.name} exceeds 5MB limit`,
      });
    }
  }

  if (totalBytes > MAX_TOTAL_BYTES) {
    return json(400, {
      error: "Total upload exceeds 50MB limit",
    });
  }

  // OCR in parallel (with concurrency cap) while preserving upload order
  let texts: string[];
  try {
    texts = await mapWithConcurrency(
      files as File[],
      OCR_CONCURRENCY,
      async (f) => {
        if (!(f instanceof File)) throw new Error("Invalid file upload");
        return await ocrSingleFile(f);
      },
    );
  } catch (e) {
    return json(500, { error: "OCR failed", detail: String((e as Error)?.message ?? e) });
  }

  const finalText = texts.filter((t) => t.length > 0).join("\n\n");
  return json(200, { text: finalText });
});