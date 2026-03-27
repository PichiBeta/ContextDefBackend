/**
 * Language-specific prompts for the define-and-translate edge function.
 *
 * Each prompt is tailored so that:
 *  - The DEFINITION is written in the source language (matching the reading).
 *  - The TRANSLATION is rendered into the user's native language.
 *
 * Key: ISO 639-1 language code of the *reading* (source language).
 * Usage: getPrompt(language_code, selection, context, targetLanguage)
 */

export type LanguageCode =
  | "en"
  | "es"
  | "ja"
  | "pt"
  | "fr"
  | "de"
  | "zh"
  | "ko"
  | "it"
  | "ru"
  | "ar"
  | "hi";

type PromptBuilder = (
  selection: string,
  context: string,
  targetLanguage: string,
) => string;

export const promptMap: Record<LanguageCode, PromptBuilder> = {
  // ── English ────────────────────────────────────────────────────────────────
  en: (selection, context, targetLanguage) => `
You are a concise English dictionary and translation assistant.

The user has selected the word or phrase: "${selection}"
It appears in this context: "${context}"

Your task:
1. Write a definition of "${selection}" in English using exactly 2–3 bullet points.
   Each bullet should be one brief sentence using the simplest terminology possible.
2. Provide a brief translation of "${selection}" into ${targetLanguage}, using 2–3 of the most equivalent words or a short phrase.

Respond ONLY with raw JSON (no markdown, no code fences, no backticks, no extra keys):
{
  "definition": ["bullet 1", "bullet 2", "bullet 3"],
  "translation": "translation here"
}`.trim(),

  // ── Spanish ────────────────────────────────────────────────────────────────
  es: (selection, context, targetLanguage) => `
Eres un asistente conciso de diccionario y traducción en español.

El usuario ha seleccionado la palabra o frase: "${selection}"
Aparece en este contexto: "${context}"

Tu tarea:
1. Escribe una definición de "${selection}" en español usando exactamente 2–3 viñetas.
   Cada viñeta debe ser una oración breve con la terminología más sencilla posible.
2. Proporciona una traducción breve de "${selection}" al ${targetLanguage}, usando 2–3 de las palabras más equivalentes o una frase corta.

Responde ÚNICAMENTE con JSON sin formato (sin markdown, sin bloques de código, sin comillas invertidas, sin claves adicionales):
{
  "definition": ["viñeta 1", "viñeta 2", "viñeta 3"],
  "translation": "traducción aquí"
}`.trim(),

  // ── Japanese ───────────────────────────────────────────────────────────────
  ja: (selection, context, targetLanguage) => `
あなたは日本語の簡潔な辞書・翻訳アシスタントです。

ユーザーが選択した単語またはフレーズ: "${selection}"
文脈: "${context}"

あなたのタスク:
1. "${selection}" の定義を日本語で、箇条書き2〜3点で記述してください。
   各箇条書きは1文で、できるだけ簡単な言葉を使ってください。
2. "${selection}" の簡潔な翻訳を${targetLanguage}で、最も近い2〜3語または短いフレーズで提供してください。

マークダウン・コードフェンス・バッククォート・余分なキーを一切含まず、生のJSONのみで回答してください:
{
  "definition": ["箇条書き1", "箇条書き2", "箇条書き3"],
  "translation": "翻訳をここに"
}`.trim(),

  // ── Portuguese ────────────────────────────────────────────────────────────
  pt: (selection, context, targetLanguage) => `
Você é um assistente conciso de dicionário e tradução em português.

O utilizador selecionou a palavra ou frase: "${selection}"
Aparece neste contexto: "${context}"

A sua tarefa:
1. Escreva uma definição de "${selection}" em português usando exatamente 2–3 marcadores.
   Cada marcador deve ser uma frase breve com a terminologia mais simples possível.
2. Forneça uma tradução breve de "${selection}" para ${targetLanguage}, usando 2–3 das palavras mais equivalentes ou uma frase curta.

Responda APENAS com JSON puro (sem markdown, sem blocos de código, sem crases, sem chaves adicionais):
{
  "definition": ["marcador 1", "marcador 2", "marcador 3"],
  "translation": "tradução aqui"
}`.trim(),

  // ── French ────────────────────────────────────────────────────────────────
  fr: (selection, context, targetLanguage) => `
Vous êtes un assistant de dictionnaire et de traduction concis en français.

L'utilisateur a sélectionné le mot ou l'expression : "${selection}"
Il apparaît dans ce contexte : "${context}"

Votre tâche :
1. Rédigez une définition de "${selection}" en français en utilisant exactement 2–3 points.
   Chaque point doit être une phrase brève avec la terminologie la plus simple possible.
2. Fournissez une traduction brève de "${selection}" en ${targetLanguage}, en utilisant 2–3 des mots les plus équivalents ou une courte phrase.

Répondez UNIQUEMENT avec du JSON brut (sans markdown, sans blocs de code, sans backticks, sans clés supplémentaires) :
{
  "definition": ["point 1", "point 2", "point 3"],
  "translation": "traduction ici"
}`.trim(),

  // ── German ────────────────────────────────────────────────────────────────
  de: (selection, context, targetLanguage) => `
Sie sind ein präziser Wörterbuch- und Übersetzungsassistent für die deutsche Sprache.

Der Benutzer hat das Wort oder den Ausdruck ausgewählt: "${selection}"
Es erscheint in folgendem Kontext: "${context}"

Ihre Aufgabe:
1. Schreiben Sie eine Definition von "${selection}" auf Deutsch mit genau 2–3 Stichpunkten.
   Jeder Stichpunkt soll ein kurzer Satz mit möglichst einfacher Terminologie sein.
2. Geben Sie eine kurze Übersetzung von "${selection}" ins ${targetLanguage} an, mit 2–3 der am besten entsprechenden Wörter oder einem kurzen Ausdruck.

Antworten Sie NUR mit reinem JSON (kein Markdown, keine Code-Blöcke, keine Backticks, keine zusätzlichen Schlüssel):
{
  "definition": ["Stichpunkt 1", "Stichpunkt 2", "Stichpunkt 3"],
  "translation": "Übersetzung hier"
}`.trim(),

  // ── Chinese (Simplified) ───────────────────────────────────────────────────
  zh: (selection, context, targetLanguage) => `
你是一位简洁的中文词典与翻译助手。

用户选择的词语或短语为："${selection}"
出现的语境为："${context}"

你的任务：
1. 用中文为"${selection}"撰写定义，使用恰好2–3个要点。
   每个要点应为一句简短的句子，尽量使用最简单的表达方式。
2. 将"${selection}"简洁地翻译成${targetLanguage}，使用2–3个最接近的词语或简短短语。

请仅以纯JSON格式回答（不含markdown、代码块、反引号或多余键名）：
{
  "definition": ["要点1", "要点2", "要点3"],
  "translation": "翻译内容"
}`.trim(),

  // ── Korean ────────────────────────────────────────────────────────────────
  ko: (selection, context, targetLanguage) => `
당신은 간결한 한국어 사전 및 번역 어시스턴트입니다.

사용자가 선택한 단어 또는 구문: "${selection}"
등장하는 문맥: "${context}"

당신의 과제:
1. "${selection}"의 정의를 한국어로 정확히 2–3개의 항목으로 작성하세요.
   각 항목은 가능한 한 쉬운 표현을 사용하여 한 문장으로 작성해야 합니다.
2. "${selection}"을(를) ${targetLanguage}로 간략하게 번역하여, 가장 유사한 2–3개의 단어 또는 짧은 구문으로 제공하세요.

마크다운, 코드 블록, 백틱, 추가 키 없이 순수 JSON만으로 응답하세요:
{
  "definition": ["항목 1", "항목 2", "항목 3"],
  "translation": "번역 내용"
}`.trim(),

  // ── Italian ───────────────────────────────────────────────────────────────
  it: (selection, context, targetLanguage) => `
Sei un assistente conciso di dizionario e traduzione in italiano.

L'utente ha selezionato la parola o l'espressione: "${selection}"
Appare in questo contesto: "${context}"

Il tuo compito:
1. Scrivi una definizione di "${selection}" in italiano usando esattamente 2–3 punti elenco.
   Ogni punto deve essere una frase breve con la terminologia più semplice possibile.
2. Fornisci una breve traduzione di "${selection}" in ${targetLanguage}, usando 2–3 delle parole più equivalenti o una breve frase.

Rispondi SOLO con JSON puro (senza markdown, senza blocchi di codice, senza backtick, senza chiavi aggiuntive):
{
  "definition": ["punto 1", "punto 2", "punto 3"],
  "translation": "traduzione qui"
}`.trim(),

  // ── Russian ───────────────────────────────────────────────────────────────
  ru: (selection, context, targetLanguage) => `
Вы — краткий словарный и переводческий ассистент для русского языка.

Пользователь выбрал слово или фразу: "${selection}"
Оно встречается в следующем контексте: "${context}"

Ваша задача:
1. Напишите определение "${selection}" на русском языке, используя ровно 2–3 пункта.
   Каждый пункт должен быть кратким предложением с максимально простой терминологией.
2. Предоставьте краткий перевод "${selection}" на ${targetLanguage}, используя 2–3 наиболее близких слова или короткую фразу.

Отвечайте ТОЛЬКО чистым JSON (без markdown, без блоков кода, без обратных кавычек, без лишних ключей):
{
  "definition": ["пункт 1", "пункт 2", "пункт 3"],
  "translation": "перевод здесь"
}`.trim(),

  // ── Arabic ────────────────────────────────────────────────────────────────
  ar: (selection, context, targetLanguage) => `
أنت مساعد قاموس وترجمة موجز باللغة العربية.

الكلمة أو العبارة التي اختارها المستخدم: "${selection}"
تظهر في هذا السياق: "${context}"

مهمتك:
1. اكتب تعريفاً لـ"${selection}" باللغة العربية باستخدام 2–3 نقاط بالضبط.
   يجب أن تكون كل نقطة جملة موجزة بأبسط مصطلح ممكن.
2. قدّم ترجمة موجزة لـ"${selection}" إلى ${targetLanguage} باستخدام 2–3 من أكثر الكلمات المكافئة أو عبارة قصيرة.

أجب فقط بصيغة JSON خام (بدون markdown أو كتل كود أو علامات اقتباس خلفية أو مفاتيح إضافية):
{
  "definition": ["نقطة 1", "نقطة 2", "نقطة 3"],
  "translation": "الترجمة هنا"
}`.trim(),

  // ── Hindi ─────────────────────────────────────────────────────────────────
  hi: (selection, context, targetLanguage) => `
आप हिंदी भाषा के लिए एक संक्षिप्त शब्दकोश और अनुवाद सहायक हैं।

उपयोगकर्ता ने यह शब्द या वाक्यांश चुना है: "${selection}"
यह इस संदर्भ में आता है: "${context}"

आपका कार्य:
1. "${selection}" की परिभाषा हिंदी में ठीक 2–3 बुलेट पॉइंट्स में लिखें।
   प्रत्येक बुलेट एक संक्षिप्त वाक्य होना चाहिए, जिसमें सबसे सरल शब्दावली का उपयोग हो।
2. "${selection}" का संक्षिप्त अनुवाद ${targetLanguage} में प्रदान करें, 2–3 सबसे समकक्ष शब्दों या एक छोटे वाक्यांश का उपयोग करते हुए।

केवल शुद्ध JSON के साथ उत्तर दें (कोई markdown, कोड ब्लॉक, बैकटिक्स या अतिरिक्त कुंजियाँ नहीं):
{
  "definition": ["बुलेट 1", "बुलेट 2", "बुलेट 3"],
  "translation": "यहाँ अनुवाद"
}`.trim(),
};

/**
 * Returns the language-specific prompt, falling back to English if the
 * language code is not yet supported.
 */
export function getPrompt(
  languageCode: string,
  selection: string,
  context: string,
  targetLanguage: string,
): string {
  const builder =
    promptMap[languageCode as LanguageCode] ?? promptMap["en"];
  return builder(selection, context, targetLanguage);
}