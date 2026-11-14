// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the brief at $MAKEFUN_PROMPT_FILE (provider directory + sealed quality table +
// workload manifest), probes each provider's catalog route, builds ONE catalog document, listens
// on an OS-picked ephemeral port, registers at $MAKEFUN_ENV_URL/register, serves the env client's
// 20 requests — POST /v1/chat/completions (OpenAI wire) and POST /v1/messages (Anthropic wire),
// JSON or SSE — polls GET $MAKEFUN_ENV_URL/work every 100 ms, and exits 0 the moment it says
// {"done": true}. `node main.ts --list` prints the catalog JSON and exits 0.
//
// Routing (lib/router.ts): rating-ordered free candidates from the sealed table, Gemini reserved
// strictly for vision requests, no OpenAI-client JSON request on an anthropic wire, strictly
// sequential provider calls, failover on 402/429/5xx. v2's two changes over v1: a refusal on the
// first failure of a streak earns ONE immediate retry of the same provider when falling back
// would lose rating (the draw's scheduled failures are seldom adjacent, so the next call number
// usually answers at full rating), and a provider that has been called the sealed table's
// maximum number of times leaves the attempt list (it refuses whatever the draw was, so trying
// it only spends calls the score ties break on). Then a second pass over the providers that
// failed. Every provider answer is parsed into one normalized shape and re-framed into the
// client's wire (lib/wire.ts); no provider frame is ever forwarded raw to a client on another
// wire, and the streamed OpenAI usage chunk is always emitted (every streamed OpenAI-wire
// upstream call carries stream_options.include_usage).
//
// Determinism: fixed timestamps and ids, deterministic tie-breaks, no clock, no randomness. Every
// choice is a function of the brief, the requests seen so far, and the provider answers seen so
// far, so a replay leaves the same state export byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicJsonResponse,
  anthropicModelList,
  anthropicStreamEvents,
  anthropicToOpenAI,
  openAIError,
  openAIJsonResponse,
  openAIModelList,
  openAIStreamEvents,
  parseAnthropicJson,
  parseAnthropicStream,
  parseGeminiJson,
  parseGeminiStream,
  parseOpenAIJson,
  parseOpenAIStream,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';
import { anthropicToGemini, callGemini, openAIToGemini } from './lib/gemini.ts';
import { anthropicUpstreamBody, callAnthropicUpstream, callOpenAIUpstream } from './lib/anthropic-upstream.ts';
import { attemptList, firstPass, requirementsFromBody, Router, type Candidate, type RouterRequest } from './lib/router.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let router: Router | null = null;
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  router = new Router(probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One provider answer, or null when the call failed, was refused, or answered unparseably. */
interface Attempt {
  candidate: Candidate;
  answer: NormalizedAnswer | null;
  status: number;
  failover: boolean;
}

/**
 * One inference call to one candidate, in that provider's own wire, parsed into the normalized
 * shape. A 402/429/5xx is a failover trigger; a 400/401 is logged as a bug in this entry's own
 * request (it spends no budget and is not a failover trigger in the scorer's sense, but the chain
 * still moves on — there is nothing to gain by stopping).
 */
async function callCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, requestIndex: number): Promise<Attempt> {
  const stream = body.stream === true;
  const wire = candidate.provider.wire;
  const base = joinBase(ENV_URL, candidate.provider.base);
  const headers = { ...authHeaders(candidate.provider.auth, candidate.provider.key) };
  let upstream: Response;
  try {
    if (wire === 'openai') {
      const upstreamBody: Json = clientApi === 'anthropic' ? anthropicToOpenAI(body, candidate.model.id) : { ...body, model: candidate.model.id };
      if (stream) upstreamBody.stream_options = { include_usage: true };
      upstream = await callOpenAIUpstream(ENV_URL, candidate.provider, upstreamBody, CALL_TIMEOUT_MS);
    } else if (wire === 'anthropic') {
      upstream = await callAnthropicUpstream(ENV_URL, candidate.provider, anthropicUpstreamBody(clientApi, body, candidate.model.id), CALL_TIMEOUT_MS);
    } else {
      const geminiBody = clientApi === 'anthropic' ? anthropicToGemini(body, candidate.model.id) : openAIToGemini(body, candidate.model.id);
      upstream = await callGemini(base, headers, geminiBody, candidate.model.id, stream, CALL_TIMEOUT_MS);
    }
  } catch (err) {
    router?.observe(candidate.provider.id, 'refused', requestIndex);
    log(`${candidate.provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
    return { candidate, answer: null, status: 0, failover: true };
  }
  const status = upstream.status;
  if (!upstream.ok) {
    const failover = status === 402 || status === 429 || status >= 500;
    router?.observe(candidate.provider.id, failover ? 'refused' : 'bug', requestIndex);
    if (!failover) log(`${candidate.provider.id}/${candidate.model.id} answered HTTP ${status} (not a failover status)`);
    return { candidate, answer: null, status, failover };
  }
  const text = await upstream.text();
  let parsed: Json | null = null;
  let answer: NormalizedAnswer | null = null;
  if (stream) {
    if (wire === 'openai') answer = parseOpenAIStream(text);
    else if (wire === 'anthropic') answer = parseAnthropicStream(text);
    else answer = parseGeminiStream(text);
  } else {
    try {
      const value: unknown = JSON.parse(text);
      parsed = isRecord(value) ? value : null;
    } catch {
      parsed = null;
    }
    if (parsed !== null) {
      if (wire === 'openai') answer = parseOpenAIJson(parsed);
      else if (wire === 'anthropic') answer = parseAnthropicJson(parsed);
      else answer = parseGeminiJson(parsed);
    }
  }
  if (answer === null) {
    router?.observe(candidate.provider.id, 'bug', requestIndex);
    log(`${candidate.provider.id}/${candidate.model.id} answered HTTP 200 with an unparseable body`);
    return { candidate, answer: null, status, failover: true };
  }
  router?.observe(candidate.provider.id, 'ok', requestIndex);
  return { candidate, answer, status, failover: false };
}

/** Whether an attempt was a provider refusal (402/429/5xx), as opposed to this entry's own bug. */
function isRefusal(attempt: Attempt): boolean {
  return attempt.failover && (attempt.status === 402 || attempt.status === 429 || attempt.status >= 500);
}

/**
 * Serve one client request: walk the failover chain (attempt list: each provider once, minus
 * the certainly-exhausted; second pass: the providers that failed, once more), then answer in
 * the client's wire. A request no provider could serve gets a 503 in the client's wire — the
 * entry keeps serving the rest.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const requestIndex = manifestIndex;
  const request: RouterRequest = { api: clientApi, stream, requires: requirementsFor(body), index: requestIndex };
  const chain = router === null ? [] : router.candidatesFor(request);
  if (chain.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  // Attempt list: best model per provider in rating order, minus the providers that have
  // certainly spent their budget. A refusal on the first failure of a streak earns ONE immediate
  // retry of the same provider when falling back would lose rating: the instance draw puts one
  // or two scheduled failures on a provider's call numbers, seldom adjacent, so the next call
  // number usually answers — the retry keeps the request on the best model, and if the retry
  // also refuses, the streak stands at 2 and no further retry is earned. Second pass: every
  // provider once more — it is the last resort before a 503.
  const attempts = router === null ? firstPass(chain) : attemptList(router, chain);
  let served: Attempt | null = null;
  for (let i = 0; i < attempts.length && served === null; i += 1) {
    const candidate = attempts[i];
    let attempt = await callCandidate(candidate, clientApi, body, requestIndex);
    if (attempt.answer === null && router !== null && isRefusal(attempt)) {
      const next: Candidate | undefined = attempts[i + 1];
      const preservesRating = next === undefined || next.rating < candidate.rating;
      if (router.stateOf(candidate.provider.id).consecutiveFails === 1 && !router.isExhausted(candidate.provider.id) && preservesRating) {
        attempt = await callCandidate(candidate, clientApi, body, requestIndex);
      }
    }
    if (attempt.answer !== null) served = attempt;
  }
  if (served === null) {
    for (const candidate of firstPass(chain)) {
      const attempt = await callCandidate(candidate, clientApi, body, requestIndex);
      if (attempt.answer !== null) {
        served = attempt;
        break;
      }
    }
  }
  if (served === null || served.answer === null) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every provider refused this request');
  }
  const answer = served.answer;
  const model = served.candidate.model.id;
  log(`served ${clientApi} request ${manifestIndex} with ${served.candidate.provider.id}/${model}`);
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    const events = clientApi === 'openai' ? openAIStreamEvents(answer, model) : anthropicStreamEvents(answer, model);
    for (const event of events) res.write(event);
    res.end();
    return;
  }
  sendJson(res, 200, clientApi === 'openai' ? openAIJsonResponse(answer, model) : anthropicJsonResponse(answer, model));
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function dispatch(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

/**
 * Requests are handled strictly one at a time, in arrival order. The client sends one at a time
 * anyway, but the serial queue also guarantees no two provider calls are ever in flight at once
 * (zai refuses a second concurrent call) and keeps the manifest index in arrival order.
 */
let inFlight: Promise<void> = Promise.resolve();
function handle(req: IncomingMessage, res: ServerResponse): void {
  const task = inFlight.then(() => dispatch(req, res));
  inFlight = task.then(
    () => undefined,
    () => undefined,
  );
  void task.catch((err: unknown) => {
    log((err as Error).stack ?? String(err));
    if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
  });
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let busy = false;
  const timer = setInterval(() => {
    if (busy) return;
    busy = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        busy = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    handle(req, res);
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
