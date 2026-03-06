# ContextDefBackend

## Pretrained Model (BETO)

This project uses the pretrained **BETO model**: `dccuchile/bert-base-spanish-wwm-cased` ([Hugging Face link](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)).

- **License**: CC BY 4.0  
- **Citation / Disclaimer**:  
  The license CC BY 4.0 best describes the use of the BETO model itself.  
  However, the original datasets used to pretrain BETO may have licenses that are **not necessarily compatible with CC BY 4.0**, especially for commercial use. Users should verify the licenses of the original text resources to ensure compliance with their intended use.  

Please provide proper attribution when using the BETO model in research or commercial projects.

## Before Running Sync

1. Install Supabase CLI (`supabase --version` should work).
2. Authenticate once: `supabase login`.
3. Ensure project targeting is set:
   - either link this repo: `supabase link --project-ref <your_project_ref>`
   - or always pass `-ProjectRef <your_project_ref>` to the script.
4. Run from repo root (`ContextDefBackend`), where `supabase/config.toml` exists.
5. If using `-WriteLocalEnvFromHost`, export/set required secret env vars on your machine first.

## Supabase Sync Script

This repo includes a one-command sync script for pulling the remote Supabase DB schema and edge functions:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync-supabase.ps1
```

Optional flags:

```powershell
# Use a specific migration name
powershell -ExecutionPolicy Bypass -File .\scripts\sync-supabase.ps1 -MigrationName 20260306190000_remote_schema

# Target a specific project ref
powershell -ExecutionPolicy Bypass -File .\scripts\sync-supabase.ps1 -ProjectRef your_project_ref

# Run only one side of the sync
powershell -ExecutionPolicy Bypass -File .\scripts\sync-supabase.ps1 -SkipDbPull
powershell -ExecutionPolicy Bypass -File .\scripts\sync-supabase.ps1 -SkipFunctionsDownload

# Generate a local secrets template from remote edge secret names + vault names found in migrations
powershell -ExecutionPolicy Bypass -File .\scripts\sync-supabase.ps1 -SyncSecretsTemplate

# Build supabase/.env.local from host environment variables for discovered secret names
powershell -ExecutionPolicy Bypass -File .\scripts\sync-supabase.ps1 -WriteLocalEnvFromHost
```

Note: Supabase only returns secret names/digests via CLI, not plaintext values.  
`-WriteLocalEnvFromHost` copies values from your machine environment variables when available.
