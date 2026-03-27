import { assertEquals, assertThrows } from "@std/assert";
import { tokenizeReading } from "../../functions/process-reading/tokenize-reading.ts";

Deno.test("tokenizeReading: schema field is 'reading_structure_v1'", () => {
  const result = tokenizeReading({ text: "Hello world.", language_code: "en" });
  assertEquals(result.schema, "reading_structure_v1");
});

Deno.test("tokenizeReading: returns sentences array", () => {
  const result = tokenizeReading({
    text: "The cat sat. The dog ran.",
    language_code: "en",
  });
  assertEquals(Array.isArray(result.sentences), true);
  assertEquals(result.sentences.length >= 1, true);
});

Deno.test("tokenizeReading: sentence spans have start < end", () => {
  const result = tokenizeReading({
    text: "Hello world. How are you? I am fine.",
    language_code: "en",
  });
  for (const s of result.sentences) {
    assertEquals(
      s.start < s.end,
      true,
      `Sentence start (${s.start}) should be < end (${s.end})`,
    );
  }
});

Deno.test("tokenizeReading: paragraphs split on double newline", () => {
  const result = tokenizeReading({
    text: "First paragraph here.\n\nSecond paragraph here.",
    language_code: "en",
  });
  assertEquals(result.blocks.length, 2, "Should produce 2 paragraph blocks");
});

Deno.test("tokenizeReading: single paragraph when no double newline", () => {
  const result = tokenizeReading({
    text: "Just one paragraph.\nWith a single newline.",
    language_code: "en",
  });
  assertEquals(result.blocks.length, 1, "Should produce 1 paragraph block");
});

Deno.test("tokenizeReading: codepoint offsets are correct for ASCII text", () => {
  const text = "Hello.";
  const result = tokenizeReading({ text, language_code: "en" });
  // The first (and only) sentence should span the whole text
  const sentence = result.sentences[0];
  assertEquals(sentence.start, 0);
  // Codepoint length of "Hello." is 6
  assertEquals(sentence.end, 6);
});

Deno.test("tokenizeReading: codepoint offsets for emoji (multi code-unit char)", () => {
  // "Hi 😀." — "😀" is 2 code units but 1 codepoint
  const text = "Hi 😀.";
  const result = tokenizeReading({ text, language_code: "en" });
  const sentence = result.sentences[0];
  assertEquals(sentence.start, 0);
  // Codepoints: H(0) i(1) (space)(2) 😀(3) .(4) => end = 5
  assertEquals(sentence.end, 5, "Emoji should count as 1 codepoint, not 2 code units");
});

Deno.test("tokenizeReading: tokens array is non-empty for normal text", () => {
  const result = tokenizeReading({ text: "The cat sat on the mat.", language_code: "en" });
  assertEquals(Array.isArray(result.tokens), true);
  assertEquals(result.tokens.length > 0, true);
});

Deno.test("tokenizeReading: throws on empty text", () => {
  assertThrows(
    () => tokenizeReading({ text: "", language_code: "en" }),
    Error,
    "tokenizeReading: text is empty",
  );
});

Deno.test("tokenizeReading: injects default language_code 'en' when missing", () => {
  const result = tokenizeReading({ text: "Hello world.", language_code: "" });
  assertEquals(result.language_code, "en");
});
