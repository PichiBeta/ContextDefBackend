import type { ProcessingContext, DifficultyResult } from "./types.ts";

/*
v1.0 - Classic readability: single scalar difficulty score derived
from Flesch–Kincaid Grade Level. Works only for English text. Normalized 0-100

Ideal version: more nuanced difficulty score, compatible with all languages the app supports
*/

function countSyllables(word: string): number {
  const w = word
    .toLowerCase()
    .replace(/[^a-z]/g, "")
    .trim();

  if (!w) return 0;
  if (w.length <= 3) return 1;

  const noSilentE = w.replace(/e\b/, "");
  const vowelGroups = noSilentE.match(/[aeiouy]+/g);
  let count = vowelGroups ? vowelGroups.length : 1;

  if (/(ia|io|eo)$/.test(w)) count += 1;
  if (/(tion|sion)$/.test(w)) count -= 1;

  return Math.max(1, count);
}

function tokenizeSentences(text: string): string[] {
  const sentences = text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);

  return sentences.length ? sentences : [text.trim()].filter(Boolean);
}

function tokenizeWords(text: string): string[] {
  return text
    .toLowerCase()
    .match(/[a-z]+(?:'[a-z]+)?/g) ?? [];
}

function computeDifficultyScore(text: string): number {
  const sentences = tokenizeSentences(text);
  const words = tokenizeWords(text);

  const sentenceCount = sentences.length || 1;
  const wordCount = words.length || 1;

  let syllableCount = 0;
  for (const w of words) syllableCount += countSyllables(w);

  const wordsPerSentence = wordCount / sentenceCount;
  const syllablesPerWord = syllableCount / wordCount;

  // Flesch–Kincaid Grade Level
  const fleschKincaidGrade =
    0.39 * wordsPerSentence + 11.8 * syllablesPerWord - 15.59;

  // Normalize: 0 = easy, 100 = hard (cap at grade 30)
  return Math.min(100, Math.max(0, (fleschKincaidGrade / 30) * 100));
}

export function calculateDifficulty(ctx: ProcessingContext): DifficultyResult {
  const text = ctx.text ?? "";
  if (!text.trim()) {
    throw new Error("calculateDifficulty: text is empty");
  }

  const score = computeDifficultyScore(text);
  return { score };
}
