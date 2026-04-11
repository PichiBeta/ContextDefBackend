import { createClient } from "supabase";
import { requireWebhookSecret } from "../_shared/auth.ts";

/*
  delete-reading-storage
  ----------------------
  Triggered by the readings_enqueue_storage_cleanup DB trigger when a reading
  row is hard-deleted from public.readings.

  Payload:  { reading_id: string }
  Auth:     x-webhook-secret (READINGS_DIFFICULTY_WEBHOOK_SECRET from Vault)
  Action:   Lists all objects in the "readings" bucket whose name starts with
            {reading_id} (catches the raw file and any derivative files such
            as *.structure.v1.json), then removes them all in a single batch
            delete call.
  Returns:  { ok: true, deleted_paths: string[] } on success
            { ok: false, detail: string }          on error (HTTP 500)
*/

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const WEBHOOK_SECRET = Deno.env.get("READINGS_DIFFICULTY_WEBHOOK_SECRET");

const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY);

Deno.serve(async (req) => {
  try {
    // 1) Authenticate
    const authResult = requireWebhookSecret(req, WEBHOOK_SECRET);
    if (authResult instanceof Response) return authResult;

    // 2) Parse payload
    const body = await req.json();
    if (typeof body.reading_id !== "string" || body.reading_id.trim().length === 0) {
      throw new Error('Invalid payload: expected non-empty "reading_id"');
    }
    const reading_id = body.reading_id.trim();

    // 3) List all objects with reading_id as prefix.
    //    Known files:
    //      {reading_id}                    — raw text (no extension)
    //      {reading_id}.structure.v1.json  — processed structure
    //    Using list() with prefix covers both and any future derivatives.
    const { data: objects, error: listErr } = await supabase.storage
      .from("readings")
      .list("", { search: reading_id });

    if (listErr) throw listErr;

    const paths = (objects ?? [])
      .filter((obj) => obj.name.startsWith(reading_id))
      .map((obj) => obj.name);

    if (paths.length === 0) {
      // Files may already be absent (e.g. upload never completed). Not an error.
      return new Response(
        JSON.stringify({ ok: true, deleted_paths: [] }),
        { headers: { "Content-Type": "application/json" } },
      );
    }

    // 4) Batch delete
    const { error: removeErr } = await supabase.storage.from("readings").remove(paths);
    if (removeErr) throw removeErr;

    console.log(
      `[delete-reading-storage] Deleted ${paths.length} file(s) for reading ${reading_id}:`,
      paths,
    );

    return new Response(
      JSON.stringify({ ok: true, deleted_paths: paths }),
      { headers: { "Content-Type": "application/json" } },
    );
  } catch (err) {
    console.error("[delete-reading-storage] Error:", (err as Error)?.message ?? err);
    return new Response(
      JSON.stringify({ ok: false, detail: String((err as Error)?.message ?? err) }),
      { status: 500, headers: { "Content-Type": "application/json" } },
    );
  }
});
