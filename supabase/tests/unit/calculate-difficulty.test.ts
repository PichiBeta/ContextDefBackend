import { assertEquals, assertThrows } from "@std/assert";
import { calculateDifficulty } from "../../functions/process-reading/calculate-difficulty.ts";

Deno.test("calculateDifficulty: returns score in [0, 100] range", () => {
  const result = calculateDifficulty({
    text: "The cat sat on the mat. The dog ran in the park.",
    language_code: "en",
  });
  const score = result.score;
  assertEquals(typeof score, "number");
  assertEquals(score >= 0, true, `Score ${score} should be >= 0`);
  assertEquals(score <= 100, true, `Score ${score} should be <= 100`);
});

Deno.test("calculateDifficulty: harder text scores higher than simple text", () => {
  const simple = calculateDifficulty({
    text: "The cat sat. The dog ran. The sun is up.",
    language_code: "en",
  });
  const complex = calculateDifficulty({
    text: "The implementation of sophisticated computational methodologies necessitates comprehensive understanding of underlying algorithmic complexities. Subsequent evaluation demonstrates substantial improvement in performance characteristics.",
    language_code: "en",
  });
  assertEquals(
    complex.score > simple.score,
    true,
    `Complex text (${complex.score}) should score higher than simple text (${simple.score})`,
  );
});

Deno.test("calculateDifficulty: single sentence with short words scores low", () => {
  const result = calculateDifficulty({ text: "I am a cat.", language_code: "en" });
  assertEquals(result.score < 30, true, `Score ${result.score} should be low for very simple text`);
});

Deno.test("calculateDifficulty: throws on empty text", () => {
  assertThrows(
    () => calculateDifficulty({ text: "", language_code: "en" }),
    Error,
    "calculateDifficulty: text is empty",
  );
});

Deno.test("calculateDifficulty: throws on whitespace-only text", () => {
  assertThrows(
    () => calculateDifficulty({ text: "   \n\t  ", language_code: "en" }),
    Error,
    "calculateDifficulty: text is empty",
  );
});

Deno.test("calculateDifficulty: result has a numeric score property", () => {
  const result = calculateDifficulty({ text: "Hello world.", language_code: "en" });
  assertEquals(typeof result.score, "number");
  assertEquals(Number.isFinite(result.score), true);
});
