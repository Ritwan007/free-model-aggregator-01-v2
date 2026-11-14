// The Anthropic upstream wire (zai): build a Messages request from either client wire and call it.
//
// From an Anthropic client the body passes through with only the model replaced, so the provider
// echoes exactly what the client sent. From an OpenAI client the body is rewritten by
// openAIToAnthropic in lib/wire.ts (system prompt, images, tool calls and results, tools, stops).
// Every call carries the anthropic-version header the wire wants, and the key in x-api-key.

import { authHeaders, joinBase, type Json, type ProviderDirectoryEntry } from './providers.ts';
import { anthropicPassthrough, openAIToAnthropic } from './wire.ts';

/** One Messages body for an anthropic-wire provider, from either client wire, naming `model`. */
export function anthropicUpstreamBody(clientApi: 'openai' | 'anthropic', body: Json, model: string): Json {
  return clientApi === 'anthropic' ? anthropicPassthrough(body, model) : openAIToAnthropic(body, model);
}

/** The headers an anthropic-wire provider wants: its key, plus the version header. */
export function anthropicUpstreamHeaders(provider: ProviderDirectoryEntry): Record<string, string> {
  return { ...authHeaders(provider.auth, provider.key), 'anthropic-version': '2023-06-01' };
}

/** One Anthropic inference call. Throws on a network failure; an HTTP error is the caller's to read. */
export async function callAnthropicUpstream(
  envUrl: string,
  provider: ProviderDirectoryEntry,
  body: Json,
  timeoutMs = 5000,
): Promise<Response> {
  const base = joinBase(envUrl, provider.base);
  return fetch(`${base}/v1/messages`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      ...(body.stream === true ? { accept: 'text/event-stream' } : {}),
      ...anthropicUpstreamHeaders(provider),
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
}

/** One OpenAI-wire provider call. The caller sets the model and stream_options on the body. */
export async function callOpenAIUpstream(
  envUrl: string,
  provider: ProviderDirectoryEntry,
  body: Json,
  timeoutMs = 5000,
): Promise<Response> {
  const base = joinBase(envUrl, provider.base);
  return fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      ...(body.stream === true ? { accept: 'text/event-stream' } : {}),
      ...authHeaders(provider.auth, provider.key),
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
}
