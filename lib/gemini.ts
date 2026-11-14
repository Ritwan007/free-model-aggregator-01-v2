// The Gemini upstream wire: build generateContent / streamGenerateContent requests from either
// client wire, and call them.
//
// The provider echoes what it parsed, so every echoed field must survive the translation:
//   systemInstruction <- the system prompt (multiple system messages joined with '\n', the way the
//     OpenAI parser joins them); contents <- the turns, user text and images intact; tools <-
//     functionDeclarations in the client's order; generationConfig.maxOutputTokens <- max_tokens;
//     generationConfig.stopSequences <- stop sequences; responseMimeType <- JSON mode.
// tool_choice is not echoed by this wire and is left out.
//
// Echo guard (same as the other translators): the provider reads the user text from the LAST
// non-model content. When the turns end on a functionResponse (a tool result), a final user content
// carrying the last user text is appended so the echoed user text still matches the client's.

import { isRecord, type Json } from './providers.ts';
import { argumentsOf, textOf } from './wire.ts';

/** One image part as Gemini spells it: inlineData for a data URL, fileData for a remote URL. */
function imagePart(url: string): Json | null {
  const dataUrl = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (dataUrl !== null) return { inlineData: { mimeType: dataUrl[1], data: dataUrl[2] } };
  if (/^https?:\/\//i.test(url)) return { fileData: { fileUri: url } };
  return null;
}

/** The Gemini tools block for one client's tool definitions, in the client's order. */
function geminiTools(definitions: Json[]): Json[] {
  const declarations: Json[] = [];
  for (const tool of definitions) {
    const name = typeof tool.name === 'string' ? tool.name : '';
    if (name === '') continue;
    declarations.push({
      name,
      description: typeof tool.description === 'string' ? tool.description : '',
      parameters: isRecord(tool.parameters) ? tool.parameters : { type: 'object', properties: {} },
    });
  }
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : [];
}

/** The Gemini generationConfig for one client's caps: max tokens, stops, sampling, JSON mode. */
function generationConfig(body: Json, jsonMode: boolean): Json {
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
  else if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/**
 * Whether a Gemini content has any text part (a functionResponse-only content reads as no user text).
 */
function contentHasText(content: Json): boolean {
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.some((p) => isRecord(p) && typeof p.text === 'string');
}

/**
 * Append a final user content carrying `lastUserText` when the turns end on a functionResponse, so
 * the provider's echoed user text is the client's last user text, not ''.
 */
function guardLastContent(contents: Json[], lastUserText: string): Json[] {
  if (contents.length === 0) return contents;
  const last = contents[contents.length - 1];
  if (last.role !== 'user' || contentHasText(last)) return contents;
  return [...contents, { role: 'user', parts: [{ text: lastUserText }] }];
}

/** An OpenAI Chat Completions body as a Gemini generateContent body for `model`. */
export function openAIToGemini(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const toolNameById = new Map<string, string>();
  let lastUserText = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : '';
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const name = typeof call.function.name === 'string' ? call.function.name : '';
        const id = typeof call.id === 'string' ? call.id : '';
        if (id !== '') toolNameById.set(id, name);
        parts.push({ functionCall: { name, args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      const callId = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      contents.push({
        role: 'user',
        parts: [{ functionResponse: { name: toolNameById.get(callId) ?? callId, response: { content: text } } }],
      });
      continue;
    }
    if (role === 'user') {
      const parts: Json[] = [];
      const content = raw.content;
      if (typeof content === 'string') {
        lastUserText = content;
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        lastUserText = textOf(content);
        for (const part of content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
          else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
            const image = imagePart(part.image_url.url);
            if (image !== null) parts.push(image);
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
      continue;
    }
  }
  const definitions: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      definitions.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const out: Json = { model, contents: guardLastContent(contents, lastUserText) };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const tools = geminiTools(definitions);
  if (tools.length > 0) out.tools = tools;
  const config = generationConfig(body, format === 'json_object' || format === 'json_schema');
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic Messages body as a Gemini generateContent body for `model`. */
export function anthropicToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  let lastUserText = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : '';
    if (role === 'assistant') {
      const parts: Json[] = [];
      const content = raw.content;
      if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use' && typeof block.name === 'string') {
            parts.push({ functionCall: { name: block.name, args: isRecord(block.input) ? block.input : {} } });
          }
        }
      } else if (typeof content === 'string' && content !== '') {
        parts.push({ text: content });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'user') {
      const parts: Json[] = [];
      const content = raw.content;
      if (typeof content === 'string') {
        lastUserText = content;
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        lastUserText = textOf(content);
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'image' && isRecord(block.source)) {
            const sourceBlock = block.source;
            if (sourceBlock.type === 'base64' && typeof sourceBlock.data === 'string') {
              const media = typeof sourceBlock.media_type === 'string' ? sourceBlock.media_type : 'image/png';
              parts.push({ inlineData: { mimeType: media, data: sourceBlock.data } });
            } else if (sourceBlock.type === 'url' && typeof sourceBlock.url === 'string') {
              const image = imagePart(sourceBlock.url);
              if (image !== null) parts.push(image);
            }
          } else if (block.type === 'tool_result') {
            const callId = typeof block.tool_use_id === 'string' ? block.tool_use_id : '';
            parts.push({ functionResponse: { name: callId, response: { content: toolResultText(block.content) } } });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
      continue;
    }
  }
  const definitions: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      definitions.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  const out: Json = { model, contents: guardLastContent(contents, lastUserText) };
  const systemText = textOf(body.system);
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const tools = geminiTools(definitions);
  if (tools.length > 0) out.tools = tools;
  const config = generationConfig(body, false);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** A tool_result block's content as text, for a Gemini functionResponse. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** The Gemini route for one model: generateContent, or streamGenerateContent for a streamed call. */
export function geminiUrl(base: string, model: string, stream: boolean): string {
  return `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
}

/** One Gemini inference call. Throws on a network failure; an HTTP error is the caller's to read. */
export async function callGemini(
  base: string,
  headers: Record<string, string>,
  body: Json,
  model: string,
  stream: boolean,
  timeoutMs = 5000,
): Promise<Response> {
  return fetch(geminiUrl(base, model, stream), {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...(stream ? { accept: 'text/event-stream' } : {}), ...headers },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
}
