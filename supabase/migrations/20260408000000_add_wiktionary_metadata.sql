-- Add part_of_speech and examples fields to word lookup/saved tables
-- to support richer Wiktionary-sourced definitions.

ALTER TABLE public.words_lookup_cache
    ADD COLUMN IF NOT EXISTS part_of_speech text,
    ADD COLUMN IF NOT EXISTS examples text NOT NULL DEFAULT '[]'::text;

ALTER TABLE public.words_saved
    ADD COLUMN IF NOT EXISTS part_of_speech text,
    ADD COLUMN IF NOT EXISTS examples text NOT NULL DEFAULT '[]'::text;
