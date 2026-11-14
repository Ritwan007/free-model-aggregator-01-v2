// Routing: rating-ordered candidates, the Gemini reserve, and the failover chain.
//
// Quality is 60% of the score, so candidates are ordered by the sealed table's rating_milli,
// highest first, with fully deterministic tie-breaks (provider id, then model id). The free-flag
// filter runs FIRST — a paid model is never a candidate, however high its rating (openrouter's
// anthropic/claude-opus-5.5 at 1520 is the trap: listed, highest-rated, paid).
//
// Two hard wire rules live here, not in the caller:
//   - Gemini is reserved strictly for vision requests. It is the only provider that can serve
//     tools+vision, and the best for vision-only; a plain request served by Gemini wastes a slot
//     no other provider can replace.
//   - An OpenAI client asking for JSON mode never goes to an anthropic-wire provider: that wire
//     cannot carry response_format, so the echo would report json_mode false and the request fails.
//
// Failover (10% of the score, and it protects served/quality too): provider calls are strictly
// sequential, one at a time, globally — zai refuses a second concurrent call, and the scorer
// requires exactly one successful provider call per request. A 402/429/5xx is a failover trigger,
// never an answer. The instance draw puts one or two scheduled failures on a provider's call
// numbers (never the first, never the last), seldom adjacent, so when the best provider refuses
// on the first failure of a streak, ONE immediate retry of the same provider usually lands on a
// working call number and keeps the request at the top rating — the caller retries only when
// falling back would lose rating. A second refusal ends the matter for that request: the chain
// moves on, and only after every candidate has been tried once does a second pass retry the
// providers that failed, in rating order — their next call numbers may be past the fail schedule.
// A 400/401 is a bug in the entry's own code, not a failover trigger: it is logged and the chain
// moves on. The entry never sleeps on a 429: the schedule advances by call NUMBER, not by time,
// so waiting changes nothing.
//
// Ordering across requests is pure rating order, as in v1. A provider that failed once is not
// pushed behind lower-rated providers for the next request: its next call number usually
// succeeds (a single scheduled failure, or the second of a pair at most), and demoting it would
// route requests to 1440-rated models while a 1475-rated one sits idle — v2 measured demotion
// variants and they lost more quality than they saved calls. The one cross-request memory v2
// does keep is a hard fact, not a suspicion: a provider already called the sealed table's
// maximum number of times refuses whatever the draw was, so it leaves the attempt list and
// stops spending tie-break calls.

import {
  capabilitiesOf,
  isRecord,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type RatingsFile,
  type Requirement,
} from './providers.ts';

/** One (provider, model) pair that may serve a request, with its sealed-table rating. */
export interface Candidate {
  provider: ProviderDirectoryEntry;
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** What one provider has done so far in this trial, from its observed answers only. */
export interface ProviderState {
  calls: number;
  consecutiveFails: number;
  successes: number;
}

/**
 * The most calls each provider can ever be asked in one trial: the top of the budget range in
 * the sealed provider table (the same table the brief's ratings come from; every instance draws
 * its budget at or below this). A provider that has been called this many times is refused by
 * the environment whatever the draw was, so attempting it again spends a tie-break call for
 * nothing — such providers leave the attempt list entirely.
 */
const BUDGET_MAX: Readonly<Record<string, number>> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

/** The request now arriving, as the router sees it. */
export interface RouterRequest {
  api: 'openai' | 'anthropic';
  stream: boolean;
  requires: Requirement;
  /** The arrival index of this request (0-based). */
  index: number;
}

const byCandidate = (a: Candidate, b: Candidate): number => {
  if (a.rating !== b.rating) return b.rating - a.rating;
  if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
  if (a.model.id !== b.model.id) return a.model.id < b.model.id ? -1 : 1;
  return 0;
};

/** The router: one candidate list per request, plus per-provider observed state. */
export class Router {
  private readonly probes: readonly Probe[];
  private readonly ratings: RatingsFile | null;
  private readonly states = new Map<string, ProviderState>();

  constructor(probes: readonly Probe[], ratings: RatingsFile | null) {
    this.probes = probes;
    this.ratings = ratings;
  }

  /** The observed state of one provider (created on first use). */
  stateOf(providerId: string): ProviderState {
    let state = this.states.get(providerId);
    if (state === undefined) {
      state = { calls: 0, consecutiveFails: 0, successes: 0 };
      this.states.set(providerId, state);
    }
    return state;
  }

  /** The most calls the sealed table allows this provider in a trial (Infinity if unknown). */
  budgetMaxOf(providerId: string): number {
    return BUDGET_MAX[providerId] ?? Number.POSITIVE_INFINITY;
  }

  /**
   * Whether this provider has certainly spent its whole budget: it has already been called the
   * sealed maximum number of times, so the environment refuses it whatever the draw was.
   */
  isExhausted(providerId: string): boolean {
    return this.stateOf(providerId).calls >= this.budgetMaxOf(providerId);
  }

  /**
   * Record one provider answer.
   *   - 'ok': a success clears the failure streak.
   *   - 'refused': a 402/429/5xx (or an unreachable provider) extends the streak. The streak is
   *     what the caller's retry rule reads: a refusal that makes the streak exactly 1 earns one
   *     immediate retry (see the header); a refusal at streak 2 or more does not.
   *   - 'bug': a 400/401 or an unparseable 200 is this entry's own fault, not the provider's:
   *     it changes no state beyond the call count.
   */
  observe(providerId: string, outcome: 'ok' | 'refused' | 'bug', requestIndex: number): void {
    const state = this.stateOf(providerId);
    state.calls += 1;
    if (outcome === 'ok') {
      state.successes += 1;
      state.consecutiveFails = 0;
    } else if (outcome === 'refused') {
      state.consecutiveFails += 1;
    }
  }

  /**
   * The free candidates that meet one request, best first.
   *
   * A candidate needs a sealed-table row AND a live catalog listing; the table decides free and
   * capabilities, the listing decides the model exists. Unknown capabilities never meet a
   * requirement (see meetsRequirements).
   */
  candidatesFor(req: RouterRequest): Candidate[] {
    const out: Candidate[] = [];
    for (const probe of this.probes) {
      if (!probe.ok) continue;
      for (const model of probe.models) {
        const row = ratingOfModel(this.ratings, probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        if (!meetsRequirements(req.requires, capabilitiesOf(model, row))) continue;
        // An OpenAI client asking for JSON mode cannot ride an anthropic wire.
        if (req.api === 'openai' && req.requires.json === true && probe.provider.wire === 'anthropic') continue;
        // Gemini is reserved strictly for vision requests.
        if (probe.provider.id === 'gemini' && req.requires.vision !== true) continue;
        out.push({ provider: probe.provider, probe, model, rating: row?.rating_milli ?? 0 });
      }
    }
    out.sort(byCandidate);
    return out;
  }
}

/**
 * The first pass over a candidate list: each provider once, with its best model for the request.
 * Apart from the caller's one retry on a streak's first refusal, a second call to the same
 * provider within one request would hit its next call number — possibly the second failure of
 * a consecutive pair — so the first pass itself never spends two attempts on one provider.
 */
export function firstPass(candidates: readonly Candidate[]): Candidate[] {
  const seen = new Set<string>();
  const out: Candidate[] = [];
  for (const candidate of candidates) {
    if (seen.has(candidate.provider.id)) continue;
    seen.add(candidate.provider.id);
    out.push(candidate);
  }
  return out;
}

/**
 * The attempt list for one request: the first pass, minus the providers that have certainly
 * spent their budget (called the sealed maximum number of times — they refuse whatever the
 * draw was, so attempting them only inflates the call count the score ties break on).
 * Pure function of the observed answers, so a replay walks the same list.
 */
export function attemptList(router: Router, candidates: readonly Candidate[]): Candidate[] {
  return firstPass(candidates).filter((candidate) => !router.isExhausted(candidate.provider.id));
}

/**
 * The best model per provider for one request, for the `/v1/models` warm-up view and for logging:
 * the same rating order as the chain, one row a provider.
 */
export function bestPerProvider(candidates: readonly Candidate[]): Candidate[] {
  return firstPass(candidates);
}

/** The requirements of a request body, read off the body itself (the manifest is the first source). */
export function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}
