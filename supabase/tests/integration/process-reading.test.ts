/**
 * Integration tests for the process-reading edge function.
 *
 * These tests call the function directly with a webhook secret, bypassing
 * the DB trigger. This lets us test the processing pipeline without needing
 * the pg_net trigger to fire.
 *
 * Prerequisites:
 *   supabase start
 *   supabase functions serve
 *   supabase/.env.local must have READINGS_DIFFICULTY_WEBHOOK_SECRET and HF_API_KEY
 *
 * Run from the supabase/ directory:
 *   cd supabase && deno test tests/integration/process-reading_test.ts --allow-net --allow-env
 */

import { assertEquals } from "@std/assert";
import {
  adminClient,
  createTestUser,
  deleteTestUser,
  FUNCTIONS_URL,
} from "./helpers/client.ts";

const WEBHOOK_SECRET = Deno.env.get("READINGS_DIFFICULTY_WEBHOOK_SECRET") ?? "";
const TEST_EMAIL = "test-process-reading@example.com";
const TEST_PASSWORD = "test-password-123!";

let userId = "";
let readingId = "";

async function setup() {
  const user = await createTestUser(TEST_EMAIL, TEST_PASSWORD);
  userId = user.userId;

  const content = "The quick brown fox jumps over the lazy dog. It was a sunny day in the meadow.";
  const now = new Date().toISOString();

  // storage_path is a generated column (= id::text), so we do not insert it.
  // content_preview is varchar(70) NOT NULL.
  const { data: reading, error } = await adminClient
    .from("readings")
    .insert({
      owner_id: userId,
      title: "Integration Test Reading",
      genre: "test",
      language_code: "en",
      visibility: "private",
      status: "uploaded",
      content_preview: content.slice(0, 70),
      content_updated_at: now,
    })
    .select("id, content_updated_at")
    .single();

  if (error || !reading) {
    throw new Error(`Failed to create test reading: ${error?.message}`);
  }

  readingId = reading.id;

  // storage_path = id (generated), so we upload to the reading's id path
  const { error: storageError } = await adminClient.storage
    .from("readings")
    .upload(readingId, new Blob([content], { type: "text/plain" }), {
      upsert: true,
    });

  if (storageError) {
    throw new Error(`Failed to upload test content: ${storageError.message}`);
  }
}

async function teardown() {
  if (readingId) {
    await adminClient.storage
      .from("readings")
      .remove([readingId, `${readingId}.structure.v1.json`]);
    await adminClient.from("readings").delete().eq("id", readingId);
  }
  if (userId) await deleteTestUser(userId);
}

Deno.test({
  name: "process-reading: returns 401 without webhook secret",
  async fn() {
    const res = await fetch(`${FUNCTIONS_URL}/process-reading`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reading_id: "test" }),
    });
    assertEquals(res.status, 401);
    await res.body?.cancel();
  },
});

Deno.test({
  name: "process-reading: processes a reading and marks it as processed",
  async fn() {
    if (!WEBHOOK_SECRET) {
      console.warn("Skipping: READINGS_DIFFICULTY_WEBHOOK_SECRET not set");
      return;
    }

    await setup();
    try {
      const { data: reading } = await adminClient
        .from("readings")
        .select("content_updated_at")
        .eq("id", readingId)
        .single();

      const res = await fetch(`${FUNCTIONS_URL}/process-reading`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-webhook-secret": WEBHOOK_SECRET,
        },
        body: JSON.stringify({
          reading_id: readingId,
          storage_path: readingId,   // matches the generated storage_path column value
          language_code: "en",
          content_updated_at: reading?.content_updated_at,
        }),
      });

      const body = await res.json();
      assertEquals(res.status, 200, `Expected 200, got ${res.status}: ${JSON.stringify(body)}`);
      assertEquals(body.ok, true);
      assertEquals(typeof body.difficulty, "number");
      assertEquals(body.difficulty >= 0 && body.difficulty <= 100, true);

      // Verify DB state
      const { data: updated } = await adminClient
        .from("readings")
        .select("status, difficulty, embedding")
        .eq("id", readingId)
        .single();

      assertEquals(updated?.status, "processed");
      assertEquals(typeof updated?.difficulty, "number");
    } finally {
      await teardown();
    }
  },
});
