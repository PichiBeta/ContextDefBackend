Summarize everything accomplished in this session as plain-text markdown, formatted for use as a Git commit message and/or PR description.

Output the entire summary inside a single triple-backtick code block with no language tag, so the user can copy-paste the raw text directly. Do not render any markdown outside the block.

Structure the content inside the block as follows:

```
<Short one-line summary in imperative mood, ≤72 chars>

## What changed
- <file/component/area — what was done and why>
- ...

## Why
<1–3 sentences on the motivation or problem solved. Omit if obvious.>

## Notes
<Edge cases, follow-ups, migration steps, or reviewer callouts. Omit section if nothing to add.>
```

Rules:
- Imperative mood on the one-liner: "Add X" not "Added X"
- Bullets describe changes to code/files/config — not process or conversation
- Be specific: use file names, function names, component names, config keys
- No filler phrases like "In this session…" or "Claude helped…"
- If the session touched multiple unrelated areas, group bullets under sub-headings inside ## What changed (e.g. ### Auth, ### UI)
- Keep it tight: one-line bullets for commit use; slightly more descriptive for PR use
- If the session was short, the summary should be short — don't pad
