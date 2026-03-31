# calculate_user_embedding

Updates a user's profile embedding as a running average when they interact with a reading (e.g. saving a reading). Called externally.

Note: `process-reading` also performs an inline user embedding update for the reading's owner at the end of processing (step 8), to avoid a redundant API call for that specific path. `calculate_user_embedding` remains the entry point for all other cases.

## Behavior

1. Receives `{ user_id, reading_id }` via webhook with `x-webhook-secret` auth.
2. Fetches reading's `embedding` from `readings` table.
3. Fetches user's `embedding` and `num_vectors` from `profiles`.
4. If `num_vectors === 0`: new embedding = reading embedding.
5. Else: calls `compute_new_embedding` RPC for running average.
6. Updates `profiles` with new embedding and incremented `num_vectors`.
