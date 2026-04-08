-- Add Wiktionary metadata columns to words_lookup_cache
ALTER TABLE public.words_lookup_cache
  ADD COLUMN IF NOT EXISTS part_of_speech text,
  ADD COLUMN IF NOT EXISTS examples text NOT NULL DEFAULT '[]';

-- Add Wiktionary metadata columns to words_saved
ALTER TABLE public.words_saved
  ADD COLUMN IF NOT EXISTS part_of_speech text,
  ADD COLUMN IF NOT EXISTS examples text NOT NULL DEFAULT '[]';
