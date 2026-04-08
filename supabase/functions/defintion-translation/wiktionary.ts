/**
 * Wiktionary REST API integration.
 *
 * Fetches part-of-speech labels and source-language example sentences
 * from the English Wiktionary. Definitions are always in English
 * (Wiktionary convention) so they are NOT used as the primary definition —
 * the LLM handles that. This module provides supplementary metadata only.
 */

// ── Types ───────────────────────────────────────────────────────────────────

interface WiktionaryDefinition {
  definition: string;
  parsedExamples?: Array<{ example: string; translation?: string }>;
  examples?: string[];
}

interface WiktionaryEntry {
  partOfSpeech: string;
  language: string;
  definitions: WiktionaryDefinition[];
}

type WiktionaryResponse = Record<string, WiktionaryEntry[]>;

export interface WiktionaryResult {
  partOfSpeech: string;
  examples: Array<{ text: string; translation?: string }>;
}

// ── HTML stripping ──────────────────────────────────────────────────────────

const ENTITY_MAP: Record<string, string> = {
  "&amp;": "&",
  "&lt;": "<",
  "&gt;": ">",
  "&#39;": "'",
  "&quot;": '"',
};

function stripHtml(html: string): string {
  return html
    .replace(/<[^>]*>/g, "")
    .replace(/&(?:amp|lt|gt|#39|quot);/g, (m) => ENTITY_MAP[m] ?? m)
    .trim();
}

// ── Main fetch ──────────────────────────────────────────────────────────────

const MAX_EXAMPLES = 3;
const TIMEOUT_MS = 3000;

/**
 * Fetches Wiktionary metadata for a single word.
 *
 * Returns part of speech and up to 3 source-language examples,
 * or `null` if the word isn't found or the request fails.
 */
export async function fetchWiktionary(
  word: string,
  languageCode: string,
): Promise<WiktionaryResult | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const url = `https://en.wiktionary.org/api/rest_v1/page/definition/${encodeURIComponent(word)}`;
    const res = await fetch(url, { signal: controller.signal });

    if (!res.ok) return null;

    const data: WiktionaryResponse = await res.json();
    const entries = findEntries(data, languageCode);
    if (!entries || entries.length === 0) return null;

    const partOfSpeech = entries[0].partOfSpeech;
    const examples = collectExamples(entries);

    return { partOfSpeech, examples };
  } catch {
    // Network error, timeout, JSON parse error — all non-fatal
    return null;
  } finally {
    clearTimeout(timer);
  }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Finds entries matching the given language code in the Wiktionary response.
 *
 * The API uses ISO 639-1 codes as top-level keys (e.g. "es", "ja", "pt").
 * Chinese may appear under "zh" or a full name like "Chinese", so we check
 * both the direct key and scan entry language fields as a fallback.
 */
function findEntries(
  data: WiktionaryResponse,
  languageCode: string,
): WiktionaryEntry[] | null {
  // Direct key match (most common case)
  if (data[languageCode]?.length) {
    return data[languageCode];
  }

  // Fallback: scan all sections for a matching language name.
  // Handles edge cases like "zh" appearing under "Chinese".
  const langNameLower = LANG_CODE_TO_WIKT_NAME[languageCode];
  if (!langNameLower) return null;

  for (const entries of Object.values(data)) {
    if (
      entries.length > 0 &&
      entries[0].language.toLowerCase() === langNameLower
    ) {
      return entries;
    }
  }

  return null;
}

const LANG_CODE_TO_WIKT_NAME: Record<string, string> = {
  en: "english",
  es: "spanish",
  ja: "japanese",
  pt: "portuguese",
  fr: "french",
  de: "german",
  zh: "chinese",
  ko: "korean",
  it: "italian",
  ru: "russian",
  ar: "arabic",
  hi: "hindi",
};

/**
 * Collects up to MAX_EXAMPLES parsed examples from all entries,
 * stripping HTML from both the example text and its translation.
 */
function collectExamples(
  entries: WiktionaryEntry[],
): Array<{ text: string; translation?: string }> {
  const results: Array<{ text: string; translation?: string }> = [];

  for (const entry of entries) {
    for (const def of entry.definitions) {
      if (results.length >= MAX_EXAMPLES) return results;

      if (def.parsedExamples) {
        for (const ex of def.parsedExamples) {
          if (results.length >= MAX_EXAMPLES) return results;

          const text = stripHtml(ex.example);
          if (!text) continue;

          const translation = ex.translation
            ? stripHtml(ex.translation)
            : undefined;
          results.push(translation ? { text, translation } : { text });
        }
      }
    }
  }

  return results;
}
