# process-reading

## Internal Modules

The `process-reading/` function contains pure, I/O-free modules ideal for unit testing:

| File | Type | What it does |
|---|---|---|
| `index.ts` | Orchestrator | Webhook auth, Storage download/upload, DB update |
| `tokenize-reading.ts` | Pure fn | `tokenizeReading(ctx)` → `ReadingStructureV1` (paragraphs, sentences, tokens with codepoint offsets) |
| `calculate-difficulty.ts` | Pure fn | `calculateDifficulty(ctx)` → `{ score: number }` (Flesch-Kincaid Grade Level, normalized 0–100) |
| `calculate-reading-embedding.ts` | Async fn | `calculateReadingEmbedding(ctx)` → `{ embedding, chunks_processed }` — HuggingFace MiniLM-L12-v2 |
