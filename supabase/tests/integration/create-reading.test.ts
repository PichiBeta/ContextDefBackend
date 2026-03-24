/**
 * Integration tests for the create-reading edge function.
 *
 * Prerequisites:
 *   supabase start
 *   supabase functions serve
 *
 * Run from the supabase/ directory:
 *   cd supabase && deno test tests/integration/create-reading_test.ts --allow-net --allow-env
 */

import { assertEquals } from "@std/assert";
import {
  adminClient,
  createTestUser,
  deleteTestUser,
  FUNCTIONS_URL,
} from "./helpers/client.ts";

const TEST_EMAIL = "test-create-reading@example.com";
const TEST_PASSWORD = "test-password-123!";

let userId = "";
let accessToken = "";

async function setup() {
  const user = await createTestUser(TEST_EMAIL, TEST_PASSWORD);
  userId = user.userId;
  accessToken = user.accessToken;
}

async function teardown() {
  await adminClient
    .from("readings")
    .delete()
    .eq("owner_id", userId);
  await deleteTestUser(userId);
}

Deno.test({
  name: "create-reading: creates a reading row and returns reading_id",
  async fn() {
    await setup();
    try {
      const res = await fetch(`${FUNCTIONS_URL}/create-reading`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({
          title: "Test Reading",
          genre: "fiction",
          language_code: "en",
          visibility: "private",
          content: "The quick brown fox jumps over the lazy dog.",
        }),
      });

      assertEquals(res.status, 200, `Expected 200, got ${res.status}`);
      const body = await res.json();
      assertEquals(typeof body.reading_id, "string", "Response should contain reading_id");
      assertEquals(body.reading_id.length > 0, true);

      // Verify the DB row was created
      const { data: reading } = await adminClient
        .from("readings")
        .select("id, status, title")
        .eq("id", body.reading_id)
        .single();

      assertEquals(reading?.title, "Test Reading");
      // Status should be 'uploaded' (or 'processing'/'processed' if webhook fires quickly)
      assertEquals(
        ["uploaded", "processing", "processed"].includes(reading?.status),
        true,
        `Unexpected status: ${reading?.status}`,
      );
    } finally {
      await teardown();
    }
  },
});

Deno.test({
  name: "create-reading: returns 401 without auth token",
  async fn() {
    const res = await fetch(`${FUNCTIONS_URL}/create-reading`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: "Test",
        genre: "fiction",
        language_code: "en",
        visibility: "private",
        content: "Some content.",
      }),
    });
    assertEquals(res.status, 401);
    await res.body?.cancel();
  },
});

Deno.test({
  name: "create-reading: returns 400 for missing required fields",
  async fn() {
    await setup();
    try {
      const res = await fetch(`${FUNCTIONS_URL}/create-reading`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ title: "Missing fields" }),
      });
      assertEquals(res.status >= 400, true, `Expected error status, got ${res.status}`);
      await res.body?.cancel();
    } finally {
      await teardown();
    }
  },
});
