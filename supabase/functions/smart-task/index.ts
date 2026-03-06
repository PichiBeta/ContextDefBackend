// supabase/functions/embed-reading/index.ts
import { createClient } from "jsr:@supabase/supabase-js@2";
import { HfInference } from "npm:@huggingface/inference";
const meanPool = (vectors)=>vectors[0].map((_, i)=>vectors.reduce((sum, v)=>sum + v[i], 0) / vectors.length);
Deno.serve(async (req)=>{
  try {
    const { reading_id } = await req.json();
    if (!reading_id) {
      return new Response(JSON.stringify({
        error: "reading_id is required"
      }), {
        status: 400,
        headers: {
          "Content-Type": "application/json"
        }
      });
    }
    const supabase = createClient(Deno.env.get("SUPABASE_URL"), Deno.env.get("SUPABASE_SERVICE_ROLE_KEY"));
    const hf = new HfInference(Deno.env.get("HF_API_KEY"));
    // STEP 1: Verify reading exists and is in 'processed' status
    const { data: reading, error: readingError } = await supabase.from("readings").select("status, language_code").eq("id", reading_id).eq("is_deleted", false).single();
    if (readingError) throw new Error(`Reading not found: ${readingError.message}`);
    if (reading.status !== "processed") {
      return new Response(JSON.stringify({
        skipped: true,
        reason: `Status is '${reading.status}'`
      }), {
        status: 200,
        headers: {
          "Content-Type": "application/json"
        }
      });
    }
    // STEP 2: Fetch the JSON structure package from Storage
    const { data: jsonFile, error: jsonError } = await supabase.storage.from("readings").download(`readings/${reading_id}.structure.v1.json`);
    if (jsonError) throw new Error(`Failed to fetch JSON package: ${jsonError.message}`);
    const readingPackage = JSON.parse(await jsonFile.text());
    // STEP 3: Fetch the raw text using storage_path from the package
    const { data: textFile, error: textError } = await supabase.storage.from("readings").download(readingPackage.text.storage_path);
    if (textError) throw new Error(`Failed to fetch raw text: ${textError.message}`);
    const rawText = await textFile.text();
    // STEP 4: Extract sentence chunks using codepoint offsets
    const chunks = readingPackage.sentences.map((s)=>rawText.slice(s.start, s.end));
    if (chunks.length === 0) throw new Error("No sentences found in reading package");
    // STEP 5: Generate embeddings using HF inference client
    const getEmbedding = async (text)=>{
      const result = await hf.featureExtraction({
        model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        inputs: text
      });
      // featureExtraction returns a nested array — flatten to 1D
      return Array.isArray(result[0]) ? result[0] : result;
    };
    // Batch in groups of 10 to avoid rate limits
    const BATCH_SIZE = 10;
    const chunkEmbeddings = [];
    for(let i = 0; i < chunks.length; i += BATCH_SIZE){
      const batch = chunks.slice(i, i + BATCH_SIZE);
      const batchEmbeddings = await Promise.all(batch.map(getEmbedding));
      chunkEmbeddings.push(...batchEmbeddings);
    }
    // STEP 6: Mean pool all chunk embeddings into one reading-level vector
    const readingEmbedding = meanPool(chunkEmbeddings);
    // STEP 7: Update the reading row with the embedding
    const { error: updateError } = await supabase.from("readings").update({
      embedding: readingEmbedding
    }).eq("id", reading_id);
    if (updateError) throw new Error(`Failed to update embedding: ${updateError.message}`);
    return new Response(JSON.stringify({
      success: true,
      reading_id,
      language_code: reading.language_code,
      chunks_processed: chunks.length
    }), {
      status: 200,
      headers: {
        "Content-Type": "application/json"
      }
    });
  } catch (err) {
    return new Response(JSON.stringify({
      error: err.message
    }), {
      status: 500,
      headers: {
        "Content-Type": "application/json"
      }
    });
  }
});
