# process-reading

## Orchestration Steps

| Step | Description |
|---|---|
| 1 | Verify webhook secret |
| 2 | Parse payload (`reading_id`, `storage_path`, `language_code`, `content_updated_at`) |
| 3 | Download raw text from Storage |
| 4 | `tokenizeReading` + `calculateDifficulty` (pure) |
| 5 | Upload structure JSON to Storage |
| 6 | `calculateReadingEmbedding` → `{ embedding, chunks_processed }` |
| 7 | Single DB update: `readings` ← `{ difficulty, embedding, status: "processed", error_message: null }` with stale-check; selects `id, owner_id` |
| 8 | **Best-effort** user embedding update: fetches `profiles.embedding` + `num_vectors`, computes running average via `compute_new_embedding` RPC, writes back to `profiles`. Failure is logged but does not affect the reading's status or HTTP response. Logic migrated from `calculate_user_embedding` (now superseded). |

## Internal Modules

The `process-reading/` function contains pure, I/O-free modules ideal for unit testing:

| File | Type | What it does |
|---|---|---|
| `index.ts` | Orchestrator | Webhook auth, Storage download/upload, DB update |
| `tokenize-reading.ts` | Pure fn | `tokenizeReading(ctx)` → `ReadingStructureV1` (paragraphs, sentences, tokens with codepoint offsets) |
| `calculate-difficulty.ts` | Pure fn | `calculateDifficulty(ctx)` → `{ score: number }` (Flesch-Kincaid Grade Level, normalized 0–100) |
| `calculate-reading-embedding.ts` | Async fn | `calculateReadingEmbedding(ctx)` → `{ embedding, chunks_processed }` — HuggingFace MiniLM-L12-v2 |
