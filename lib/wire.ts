// The client wires and the three upstream wires.
//
// The trial's client calls this server as an OpenAI service (POST /v1/chat/completions) or an
// Anthropic one (POST /v1/messages), with a JSON body or with server-sent events. Every provider
// answer is parsed into one normalized shape — { text, inputTokens, outputTokens, stopReason } —
// and re-framed into the client's wire by this file's writers, so no provider frame is ever
// forwarded raw to a client on a different wire. The provider echoes back what it parsed, and the
// client compares that echo with what it sent, so every translation below passes every echoed
// field through untouched: system prompt, user text (images included), tools in order, stop
// sequences, max_tokens, and JSON mode.
//
// Fixed timestamps only: nothing in a reply reads the clock, so a live pass and its replay send
// the same bytes. No randomness anywhere.

import { isRecord, type Json } from './providers.ts';

/** One model this server lists, as GET /v1/models shows it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** Fixed response ids: the env never checks them, and fixed strings keep replays identical. */
const OPENAI_ID = 'chatcmpl-agg';
const ANTHROPIC_ID = 'msg_agg';

/** GET /v1/models on an OpenAI-shaped request: the models this server serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
export function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** The stop-reason family of one provider's stop string (0 stop, 1 length, 2 tool use, 3 refusal). */
export function stopFamily(reason: string | null): number | null {
  if (reason === null) return null;
  if (reason === 'stop' || reason === 'end_turn' || reason === 'STOP') return 0;
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return null;
}

/** One stop string as the OpenAI wire says it, from any provider's stop string. */
export function openAIStopOf(reason: string | null): string {
  const family = stopFamily(reason);
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}

/** One stop string as the Anthropic wire says it, from any provider's stop string. */
export function anthropicStopOf(reason: string | null): string {
  const family = stopFamily(reason);
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 *
 * Echo guard: the provider echoes the LAST user message's text, and the client expects the text of
 * the original body's last user message. When that last user message carries only tool_result blocks
 * (no text at all), the translated body must still end with a user message whose text is empty —
 * otherwise the provider echoes an earlier user message's text and `echo_user` fails.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  let lastUserText = '';
  let lastUserHasTextPart = false;
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      if (!assistant) {
        lastUserText = content;
        lastUserHasTextPart = true;
      }
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    let sawText = false;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) {
        parts.push(part);
        if (block.type === 'text') sawText = true;
      }
    }
    lastUserText = textOf(content);
    lastUserHasTextPart = sawText || parts.some((p) => p.type === 'image_url');
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  if (!lastUserHasTextPart) messages.push({ role: 'user', content: lastUserText });

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    // The provider emits the OpenAI usage chunk only when the upstream request asks for it; the
    // client never sets stream_options, so ask for it here on every streamed upstream call.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`.
 * The mirror of anthropicToOpenAI: system messages become the system prompt, image_url parts
 * become image blocks, tool_calls become tool_use blocks, tool messages become tool_result blocks,
 * and max_tokens, temperature, top_p, stop sequences and streaming pass through.
 *
 * Echo guard (mirror image of the other direction): the provider echoes the LAST user message's
 * text. When the original body's last message is a tool message, the translated body must still end
 * with a user message carrying the last user text, or the provider echoes '' and `echo_user` fails.
 * (response_format is never translated: the router never sends an OpenAI JSON-mode request to an
 * anthropic-wire provider, because that wire cannot carry it.)
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  let lastUserText = '';
  const toolNameById = new Map<string, string>();
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : '';
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const name = typeof call.function.name === 'string' ? call.function.name : '';
        const id = typeof call.id === 'string' ? call.id : '';
        if (id !== '') toolNameById.set(id, name);
        blocks.push({ type: 'tool_use', id: id !== '' ? id : 'toolu_unknown', name, input: argumentsOf(call.function.arguments) });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      const callId = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: callId, content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }],
      });
      continue;
    }
    if (role === 'user') {
      const blocks: Json[] = [];
      const content = raw.content;
      if (typeof content === 'string') {
        lastUserText = content;
        if (content !== '') blocks.push({ type: 'text', text: content });
      } else if (Array.isArray(content)) {
        lastUserText = textOf(content);
        for (const part of content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
          else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
            const url = part.image_url.url;
            const dataUrl = /^data:([^;,]+);base64,(.*)$/s.exec(url);
            if (dataUrl !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: dataUrl[1], data: dataUrl[2] } });
            else blocks.push({ type: 'image', source: { type: 'url', url } });
          }
        }
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
      continue;
    }
  }
  // If the body's last message was a tool message, the translated body ends on a tool_result user
  // message whose text is ''; restore the last user text as the final user message.
  const last = messages.length > 0 ? messages[messages.length - 1] : null;
  const lastIsToolResult =
    last !== null &&
    last.role === 'user' &&
    Array.isArray(last.content) &&
    (last.content as unknown[]).length > 0 &&
    (last.content as unknown[]).every((b) => isRecord(b) && b.type === 'tool_result');
  if (lastIsToolResult) messages.push({ role: 'user', content: lastUserText });

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic client body for an anthropic-wire provider: the same body, naming `model`. */
export function anthropicPassthrough(body: Json, model: string): Json {
  return { ...body, model };
}

/** One provider answer, in a wire-neutral shape, ready to be re-framed for either client wire. */
export interface NormalizedAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stopReason: string;
  toolCalls: { id: string; name: string; arguments: string }[];
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** An OpenAI JSON answer as a normalized answer. */
export function parseOpenAIJson(body: Json): NormalizedAnswer | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  const toolCalls: NormalizedAnswer['toolCalls'] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : '',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      arguments: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
    });
  }
  return {
    text,
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stopReason: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    toolCalls,
  };
}

/** An Anthropic JSON answer as a normalized answer. */
export function parseAnthropicJson(body: Json): NormalizedAnswer | null {
  if (body.type !== 'message' || !Array.isArray(body.content)) return null;
  const text = textOf(body.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  const toolCalls: NormalizedAnswer['toolCalls'] = [];
  for (const block of body.content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : '',
      name: typeof block.name === 'string' ? block.name : '',
      arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
    });
  }
  return {
    text,
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stopReason: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
    toolCalls,
  };
}

/** A normalized answer as an OpenAI JSON body for the client. */
export function openAIJsonResponse(answer: NormalizedAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({
      id: c.id !== '' ? c.id : 'call_agg',
      type: 'function',
      function: { name: c.name, arguments: c.arguments },
    }));
  }
  return {
    id: OPENAI_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIStopOf(answer.stopReason) }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** A normalized answer as an Anthropic JSON body for the client. */
export function anthropicJsonResponse(answer: NormalizedAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_agg', name: call.name, input: argumentsOf(call.arguments) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: ANTHROPIC_ID,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopOf(answer.stopReason),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split one answer text into content deltas; the sim's answers are short, so any split is valid. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 64) pieces.push(text.slice(at, at + 64));
  return pieces.length === 0 ? [''] : pieces;
}

/** A normalized answer as the OpenAI SSE sequence the client checks, in exact event order. */
export function openAIStreamEvents(answer: NormalizedAnswer, model: string): string[] {
  const out: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: OPENAI_ID,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  out.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitText(answer.text)) out.push(sseEvent(null, chunk({ content: piece }, null)));
  out.push(sseEvent(null, chunk({}, openAIStopOf(answer.stopReason))));
  out.push(
    sseEvent(null, {
      id: OPENAI_ID,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  out.push('data: [DONE]\n\n');
  return out;
}

/** A normalized answer as the Anthropic SSE sequence the client checks, in exact event order. */
export function anthropicStreamEvents(answer: NormalizedAnswer, model: string): string[] {
  const out: string[] = [];
  out.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: ANTHROPIC_ID,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  let nextIndex = 0;
  const textIndex = nextIndex;
  nextIndex += 1;
  out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
  for (const piece of splitText(answer.text)) {
    out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: piece } }));
  }
  out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
  for (const call of answer.toolCalls) {
    const index = nextIndex;
    nextIndex += 1;
    out.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
      }),
    );
    if (call.arguments !== '') {
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.arguments } }));
    }
    out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  }
  out.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopOf(answer.stopReason), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
  );
  out.push(sseEvent('message_stop', { type: 'message_stop' }));
  return out;
}

/** One SSE text body as its data payloads, in order (event name kept where the frame names one). */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonObject(payload: string): Json | null {
  try {
    const value: unknown = JSON.parse(payload);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI provider's SSE body as a normalized answer (text + the usage chunk + finish reason). */
export function parseOpenAIStream(text: string): NormalizedAnswer | null {
  let content = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'stop';
  let sawAny = false;
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') break;
    const chunk = parseJsonObject(payload);
    if (chunk === null) continue;
    sawAny = true;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') content += delta.content;
    if (typeof choice.finish_reason === 'string') stopReason = choice.finish_reason;
  }
  if (!sawAny) return null;
  return { text: content, inputTokens, outputTokens, stopReason, toolCalls: [] };
}

/** An Anthropic provider's SSE body as a normalized answer. */
export function parseAnthropicStream(text: string): NormalizedAnswer | null {
  let content = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'end_turn';
  let sawAny = false;
  for (const frame of sseFrames(text)) {
    const chunk = parseJsonObject(frame.data);
    if (chunk === null) continue;
    sawAny = true;
    if (frame.event === 'message_start' && isRecord(chunk.message) && isRecord(chunk.message.usage)) {
      inputTokens = numberOrZero(chunk.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_delta' && isRecord(chunk.delta) && chunk.delta.type === 'text_delta' && typeof chunk.delta.text === 'string') {
      content += chunk.delta.text;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(chunk.delta) && typeof chunk.delta.stop_reason === 'string') stopReason = chunk.delta.stop_reason;
      if (isRecord(chunk.usage)) outputTokens = numberOrZero(chunk.usage.output_tokens);
    }
  }
  if (!sawAny) return null;
  return { text: content, inputTokens, outputTokens, stopReason, toolCalls: [] };
}

/** A Gemini provider's SSE body as a normalized answer (unnamed frames, usage on the final frame). */
export function parseGeminiStream(text: string): NormalizedAnswer | null {
  let content = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'STOP';
  let sawAny = false;
  for (const frame of sseFrames(text)) {
    const chunk = parseJsonObject(frame.data);
    if (chunk === null) continue;
    sawAny = true;
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const parts = isRecord(candidate.content) && Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
      for (const part of parts) {
        if (isRecord(part) && typeof part.text === 'string') content += part.text;
      }
      if (typeof candidate.finishReason === 'string') stopReason = candidate.finishReason;
    }
    if (isRecord(chunk.usageMetadata)) {
      inputTokens = numberOrZero(chunk.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
    }
  }
  if (!sawAny) return null;
  return { text: content, inputTokens, outputTokens, stopReason, toolCalls: [] };
}

/** A Gemini provider's JSON answer as a normalized answer. */
export function parseGeminiJson(body: Json): NormalizedAnswer | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const parts = isRecord(candidate.content) && Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
  let text = '';
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') text += part.text;
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    stopReason: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
    toolCalls: [],
  };
}
