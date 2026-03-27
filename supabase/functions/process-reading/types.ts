export interface ProcessingContext {
  text: string;
  language_code?: string;
}

export interface DifficultyResult {
  score: number;
}

// deno-lint-ignore no-explicit-any
export type ReadingStructureV1 = any;
