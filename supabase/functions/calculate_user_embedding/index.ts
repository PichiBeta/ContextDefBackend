// supabase/functions/update-user-embedding/index.ts
import { createClient } from "jsr:@supabase/supabase-js@2";
Deno.serve(async (req)=>{
  const { user_id, reading_id } = await req.json();
  const supabase = createClient(Deno.env.get("SUPABASE_URL"), Deno.env.get("SUPABASE_SERVICE_ROLE_KEY"));
  // Fetch the reading's embedding
  const { data: reading, error: readingError } = await supabase.from("readings").select("embedding").eq("id", reading_id).single();
  if (readingError || !reading) {
    return new Response(JSON.stringify({
      error: "Reading not found"
    }), {
      status: 404
    });
  }
  if (!reading.embedding) {
    return new Response(JSON.stringify({
      error: "No embedding associated with reading"
    }), {
      status: 422
    });
  }
  // Fetch the user's current embedding and num_vectors
  const { data: user, error: userError } = await supabase.from("profiles").select("embedding, num_vectors").eq("id", user_id).single();
  if (userError || !user) {
    return new Response(JSON.stringify({
      error: "User not found"
    }), {
      status: 404
    });
  }
  let newEmbedding;
  const newNumVectors = user.num_vectors + 1;
  if (user.num_vectors === 0) {
    // First reading — just use the reading's embedding directly
    newEmbedding = reading.embedding;
  } else {
    // Compute new running average via SQL to avoid pulling full vectors into JS
    const { data: avgData, error: avgError } = await supabase.rpc("compute_new_embedding", {
      old_embedding: user.embedding,
      new_embedding: reading.embedding,
      num_vectors: user.num_vectors
    });
    if (avgError) {
      return new Response(JSON.stringify({
        error: avgError.message
      }), {
        status: 500
      });
    }
    newEmbedding = avgData;
  }
  // Update the user
  const { error: updateError } = await supabase.from("profiles").update({
    embedding: newEmbedding,
    num_vectors: newNumVectors
  }).eq("id", user_id);
  if (updateError) {
    return new Response(JSON.stringify({
      error: updateError.message
    }), {
      status: 500
    });
  }
  return new Response(JSON.stringify({
    success: true,
    num_vectors: newNumVectors
  }), {
    headers: {
      "Content-Type": "application/json"
    }
  });
});
