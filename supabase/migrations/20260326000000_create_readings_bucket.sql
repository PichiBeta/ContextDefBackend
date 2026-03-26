-- Create the "readings" storage bucket (code-first).
-- The RLS policies for this bucket are already defined in the initial migration.
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types, avif_autodetection)
VALUES (
  'readings',
  'readings',
  false,
  1048576,  -- 1 MB
  ARRAY['text/*', 'application/json'],
  false
)
ON CONFLICT (id) DO NOTHING;
