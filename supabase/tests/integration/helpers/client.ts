import { createClient } from "supabase";

/**
 * Shared Supabase clients for integration tests.
 * Reads from environment (populated from supabase/.env.local via `supabase functions serve`).
 *
 * Defaults target the local Docker stack started by `supabase start`.
 */
const SUPABASE_URL =
  Deno.env.get("SUPABASE_URL") ?? "http://127.0.0.1:54321";
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
const ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY") ?? "";

/** Admin client — bypasses RLS. Use for test setup and teardown. */
export const adminClient = createClient(SUPABASE_URL, SERVICE_ROLE_KEY);

/** Anon client — subject to RLS. Use for function calls that represent the client perspective. */
export const anonClient = createClient(SUPABASE_URL, ANON_KEY);

/** Base URL for calling edge functions via the local gateway. */
export const FUNCTIONS_URL = `${SUPABASE_URL}/functions/v1`;

/**
 * Creates a test user via the admin API and returns their session.
 * Call deleteTestUser(userId) in afterEach.
 */
export async function createTestUser(
  email: string,
  password: string,
): Promise<{ userId: string; accessToken: string }> {
  const { data, error } = await adminClient.auth.admin.createUser({
    email,
    password,
    email_confirm: true,
  });
  if (error || !data.user) {
    throw new Error(`Failed to create test user: ${error?.message}`);
  }
  const { data: session, error: signInError } = await anonClient.auth.signInWithPassword({
    email,
    password,
  });
  if (signInError || !session.session) {
    throw new Error(`Failed to sign in test user: ${signInError?.message}`);
  }
  return {
    userId: data.user.id,
    accessToken: session.session.access_token,
  };
}

/** Deletes a test user and all their data. Call in afterEach. */
export async function deleteTestUser(userId: string): Promise<void> {
  await adminClient.auth.admin.deleteUser(userId);
}
