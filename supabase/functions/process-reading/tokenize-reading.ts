// deno-lint-ignore-file no-explicit-any

import winkTokenizer from "wink-tokenizer";
import type { ProcessingContext, ReadingStructureV1 } from "./types.ts";

/**
 * Tokenization + structural packaging (pure).
 * - DOES NOT know about reading_id or storage_path (index.ts injects those).
 * - Returns a ReadingStructureV1-ish object (ReadingStructureV1 is `any` for now).
 * - Offsets are codepoint-based (implicit, as decided).
 */

// --- helpers for codepoint offsets (minimal, deterministic) ---
function buildCodeUnitToCodePointMap(text: string): number[] {
  // map[codeUnitIndex] = codePointIndex
  const map = new Array<number>(text.length + 1);
  let cu = 0;
  let cp = 0;

  for (const ch of text) {
    const nextCu = cu + ch.length; // ch.length is 1 or 2 code units
    for (let i = cu; i < nextCu; i++) map[i] = cp;
    cu = nextCu;
    cp++;
  }
  map[cu] = cp;
  return map;
}

function makeParagraphBlocks(text: string, cuToCp: number[]) {
  // Paragraph blocks split on \n\n+ (runs of blank lines).
  const blocks: { i: number; type: "paragraph"; start: number; end: number }[] =
    [];

  const n = text.length;
  let startCu = 0;
  let i = 0;

  while (i < n) {
    // detect separator: \n\n+ (two or more newlines)
    if (text[i] === "\n" && text[i + 1] === "\n") {
      const endCu = i; // exclude the newline separator from the block

      if (endCu > startCu) {
        // trim trailing whitespace within the paragraph
        let trimmedEnd = endCu;
        while (trimmedEnd > startCu && /\s/.test(text[trimmedEnd - 1])) {
          trimmedEnd--;
        }

        if (trimmedEnd > startCu) {
          blocks.push({
            i: blocks.length,
            type: "paragraph",
            start: cuToCp[startCu],
            end: cuToCp[trimmedEnd],
          });
        }
      }

      // skip all consecutive newlines
      while (i < n && text[i] === "\n") i++;
      startCu = i;
      continue;
    }

    i++;
  }

  // tail block
  if (startCu < n) {
    // trim trailing whitespace at end-of-file
    let trimmedEnd = n;
    while (trimmedEnd > startCu && /\s/.test(text[trimmedEnd - 1])) {
      trimmedEnd--;
    }

    if (trimmedEnd > startCu) {
      blocks.push({
        i: blocks.length,
        type: "paragraph",
        start: cuToCp[startCu],
        end: cuToCp[trimmedEnd],
      });
    }
  }

  return blocks;
}

function makeSentenceSpans(text: string, cuToCp: number[]) {
  // Simple rules:
  // - boundary at .?! possibly followed by closing quotes/brackets, then whitespace/newline
  // - boundary at \n\n
  const spans: { i: number; start: number; end: number }[] = [];
  const isTerm = (c: string) => c === "." || c === "?" || c === "!";
  const isCloser = (c: string) =>
    c === `"` || c === `'` || c === ")" || c === "]" || c === "}";
  const isWS = (c: string) =>
    c === " " || c === "\n" || c === "\t" || c === "\r";

  let startCu = 0;
  let i = 0;
  const n = text.length;

  while (i < n) {
    // hard paragraph break
    if (text[i] === "\n" && text[i + 1] === "\n") {
      const endCu = i; // exclude the newline whitespace from sentence
      if (endCu > startCu) {
        spans.push({
          i: spans.length,
          start: cuToCp[startCu],
          end: cuToCp[endCu],
        });
      }
      // skip consecutive newlines
      while (i < n && text[i] === "\n") i++;
      startCu = i;
      continue;
    }

    if (isTerm(text[i])) {
      let j = i + 1;
      while (j < n && isCloser(text[j])) j++;
      // boundary if followed by whitespace/newline OR end of text
      if (j >= n || isWS(text[j])) {
        const endCu = j; // include terminator + closers, exclude following whitespace
        if (endCu > startCu) {
          spans.push({
            i: spans.length,
            start: cuToCp[startCu],
            end: cuToCp[endCu],
          });
        }
        // advance through whitespace
        while (j < n && isWS(text[j])) j++;
        startCu = j;
        i = j;
        continue;
      }
    }

    i++;
  }

  // tail
  if (startCu < n) {
    spans.push({
      i: spans.length,
      start: cuToCp[startCu],
      end: cuToCp[n],
    });
  }

  return spans;
}

export function tokenizeReading(ctx: ProcessingContext): ReadingStructureV1 {
  const content = ctx.text ?? "";
  if (!content.trim()) {
    throw new Error("tokenizeReading: text is empty");
  }

  const language_code = (ctx.language_code ?? "").trim() || "en";

  const cuToCp = buildCodeUnitToCodePointMap(content);

  const tokenizer = winkTokenizer();
  const rawTokens = tokenizer.tokenize(content);

  // Build tokens with offsets by monotonic search in raw content (code units),
  // then convert to codepoint offsets via cuToCp.
  let cursorCu = 0;
  const tokens: Array<{
    i: number;
    start: number;
    end: number;
    surface: string;
    norm: string;
    kind: "word" | "number" | "punct" | "symbol" | "other";
  }> = [];

  for (const t of rawTokens) {
    const surface = String((t as any).value ?? "");
    if (!surface) continue;

    const tag = String((t as any).tag ?? "").toLowerCase();
    let kind: "word" | "number" | "punct" | "symbol" | "other" = "other";
    if (tag === "word") kind = "word";
    else if (tag === "number") kind = "number";
    else if (tag === "punctuation") kind = "punct";
    else if (tag === "symbol") kind = "symbol";

    const found = content.indexOf(surface, cursorCu);
    if (found === -1) {
      // If we can't locate it, skip rather than corrupt offsets.
      continue;
    }

    const startCu = found;
    const endCu = found + surface.length;

    tokens.push({
      i: tokens.length,
      start: cuToCp[startCu],
      end: cuToCp[endCu],
      surface,
      norm: surface.toLowerCase(),
      kind,
    });

    cursorCu = endCu;
  }

  // Sentence and paragraph spans (simple rules)
  const sentences = makeSentenceSpans(content, cuToCp);
  const blocks = makeParagraphBlocks(content, cuToCp);

  // Spans: keep empty for now (tokens-only) to keep changes minimal.
  const spans: Array<{
    id: string;
    type: "phrase" | "contraction";
    start: number;
    end: number;
    surface: string;
    token_range: [number, number];
  }> = [];

  // index.ts injects reading_id + text.storage_path.
  const readingStructure = {
    schema: "reading_structure_v1",
    reading_id: null, // index.ts should overwrite
    language_code,
    text: {
      storage_path: null, // index.ts should overwrite
      format: "text/plain",
      offset_unit: "codepoint",
    },
    processor: {
      tokenizer: "wink-tokenizer",
      tokenizer_version: "5",
      phrase_rules_version: "1.0.0",
    },
    blocks,
    sentences,
    tokens,
    spans,
  };

  return readingStructure as ReadingStructureV1;
}
