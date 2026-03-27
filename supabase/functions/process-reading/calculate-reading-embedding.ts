import { HfInference } from "huggingface-inference";

const MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2";
const BATCH_SIZE = 10;

function meanPool(vectors: number[][]): number[] {
  return vectors[0].map((_, i) =>
    vectors.reduce((sum, v) => sum + v[i], 0) / vectors.length,
  );
}

export async function calculateReadingEmbedding(ctx: {
  text: string;
  structure: { sentences?: Array<{ start: number; end: number }> };
}): Promise<{ embedding: number[]; chunks_processed: number }> {
  const text = ctx.text ?? "";
  if (!text.trim()) {
    throw new Error("calculateReadingEmbedding: text is empty");
  }

  const sentences = ctx.structure?.sentences ?? [];
  // STEP 1: Extract sentence chunks using codepoint offsets
  const chunks = sentences
    .map((s) => text.slice(s.start, s.end))
    .filter((chunk) => chunk.trim().length > 0);

  if (chunks.length === 0) {
    throw new Error("No sentences found in reading structure");
  }

  const hfApiKey = Deno.env.get("HF_API_KEY");
  if (!hfApiKey) {
    throw new Error("Missing HF_API_KEY");
  }

  const hf = new HfInference(hfApiKey);
  // STEP 2: Generate embeddings using HF inference client
  const getEmbedding = async (chunkText: string): Promise<number[]> => {
    const result = await hf.featureExtraction({
      model: MODEL,
      inputs: chunkText,
    });

    // featureExtraction returns a nested array - flatten to 1D
    return Array.isArray(result[0]) ? (result[0] as number[]) : (result as number[]);
  };

  // Batch in groups of 10 to avoid rate limits
  const chunkEmbeddings: number[][] = [];
  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE);
    const batchEmbeddings = await Promise.all(batch.map(getEmbedding));
    chunkEmbeddings.push(...batchEmbeddings);
  }

  // STEP 3: Mean pool all chunk embeddings into one reading-level vector
  return {
    embedding: meanPool(chunkEmbeddings),
    chunks_processed: chunks.length,
  };
}
