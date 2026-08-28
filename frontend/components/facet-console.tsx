'use client'

import { useState, useEffect } from 'react'
import {
  CircleCheck,
  CircleAlert,
  ChevronRight,
  Sparkles,
  X,
  Activity,
  Server,
  Cpu,
  RefreshCw,
  Quote,
  ShieldAlert,
  CheckCircle2,
  HelpCircle
} from 'lucide-react'
import {
  evaluateConversation,
  checkHealth,
  type EvaluateResponse,
  type FacetScoreResult,
  type HealthResponse
} from '@/lib/api'

// Predefined demo snippets covering all benchmark cases
const DEMO_SNIPPETS = [
  {
    label: 'Clear Evidence',
    badge: 'Scored',
    text: 'I decided to call the customer myself rather than wait for my manager.',
    desc: 'Clear observable assertiveness and direct initiative.'
  },
  {
    label: 'Ambiguous / Fearfulness',
    badge: 'Scored',
    text: 'My knees were knocking in sheer terror, but I took the stage anyway.',
    desc: 'Exhibits fearfulness while demonstrating courage under pressure.'
  },
  {
    label: 'Medical Hallucination Trap',
    badge: 'Abstained',
    text: 'I have been feeling dizzy lately and my blood pressure is 140/90.',
    desc: 'Medical biomarker / clinical metric — system abstains.'
  },
  {
    label: 'External Log Trap',
    badge: 'Abstained',
    text: 'I commute 45 minutes to work every day and own a blue sedan.',
    desc: 'External system log / biographical fact — system abstains.'
  },
  {
    label: 'Quoted / Sarcasm',
    badge: 'Complex',
    text: 'Oh sure, I LOVE staying past midnight fixing your typos for free!',
    desc: 'Sarcastic / quoted tone expressing underlying acidity.'
  }
]

// 5-Level Ordinal Scale Labels matching backend taxonomy
const SCALE_LABELS: Record<number, string> = {
  1: 'Very Low',
  2: 'Low',
  3: 'Moderate',
  4: 'High',
  5: 'Very High'
}

export default function FacetConsole() {
  const [text, setText] = useState('')
  const [result, setResult] = useState<EvaluateResponse | null>(null)
  const [selectedFacet, setSelectedFacet] = useState<FacetScoreResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingStep, setLoadingStep] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)

  // Fetch health status on mount
  useEffect(() => {
    fetchHealthStatus()
    const interval = setInterval(fetchHealthStatus, 15000)
    return () => clearInterval(interval)
  }, [])

  async function fetchHealthStatus() {
    try {
      const data = await checkHealth()
      setHealth(data)
    } catch {
      setHealth(null)
    }
  }

  async function handleEvaluate() {
    if (!text.trim() || loading) return
    setLoading(true)
    setError(null)
    setLoadingStep(1)

    // Simulate step progress indicators
    const t1 = setTimeout(() => setLoadingStep(2), 600)
    const t2 = setTimeout(() => setLoadingStep(3), 1200)

    try {
      const res = await evaluateConversation(text)
      setResult(res)
    } catch (err: any) {
      setError(err.message || 'An error occurred while communicating with the backend.')
    } finally {
      clearTimeout(t1)
      clearTimeout(t2)
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#080a0d] text-[#f3f5f7] font-sans">
      {/* Header */}
      <header className="border-b border-white/10 bg-[#0d1016]/80 backdrop-blur-md sticky top-0 z-10">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-lg border border-cyan-500/30 bg-cyan-500/10 text-cyan-400">
              <Sparkles size={18} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-semibold tracking-tight text-white">Facet Evaluator</span>
                <span className="rounded bg-cyan-500/10 px-1.5 py-0.5 text-[10px] font-mono text-cyan-400 border border-cyan-500/20">
                  v1.0.0
                </span>
              </div>
              <p className="text-xs text-slate-400">Behavioral & Semantic Facet Evaluation Engine</p>
            </div>
          </div>

          {/* System Health Indicators */}
          <div className="flex items-center gap-4 text-xs font-mono">
            <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1">
              <Server size={13} className={health ? 'text-emerald-400' : 'text-rose-400'} />
              <span className="text-slate-300">Backend:</span>
              <span className={health ? 'text-emerald-400 font-medium' : 'text-rose-400 font-medium'}>
                {health ? 'Connected' : 'Offline'}
              </span>
            </div>

            <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1">
              <Cpu size={13} className={health?.colab_inference_status === 'online' ? 'text-emerald-400' : 'text-amber-400'} />
              <span className="text-slate-300">Model:</span>
              <span className={health?.colab_inference_status === 'online' ? 'text-emerald-400 font-medium' : 'text-amber-400 font-medium'}>
                {health?.colab_inference_status === 'online' ? 'Colab Online' : 'Standby / Local'}
              </span>
            </div>

            <button
              onClick={fetchHealthStatus}
              title="Refresh Health"
              className="rounded-full p-1.5 text-slate-400 hover:bg-white/10 hover:text-white transition"
            >
              <RefreshCw size={13} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="mx-auto max-w-6xl px-6 py-8">
        {!result && !loading ? (
          /* Workflow Input View */
          <div className="mx-auto max-w-3xl pt-8">
            <div className="text-center">
              <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                Evaluate Conversational Signals
              </h1>
              <p className="mt-3 text-base text-slate-400 max-w-xl mx-auto">
                Extract grounded behavioral facets, score evidence on a 5-level scale, and enforce conservative abstention against unobservable constructs.
              </p>
            </div>

            {/* Conversation Input Box */}
            <div className="mt-8">
              <div className="overflow-hidden rounded-xl border border-white/10 bg-[#101318] focus-within:border-cyan-500/50 focus-within:ring-1 focus-within:ring-cyan-500/50 transition">
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Paste or type a conversation dialogue transcript here..."
                  className="min-h-44 w-full resize-none bg-transparent p-5 text-sm text-slate-200 placeholder-slate-500 outline-none leading-relaxed"
                />

                <div className="flex items-center justify-between border-t border-white/10 bg-white/[0.02] px-5 py-3">
                  <span className="font-mono text-xs text-slate-400">{text.length} characters</span>
                  <button
                    onClick={handleEvaluate}
                    disabled={!text.trim() || loading}
                    className="inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-5 py-2 text-sm font-medium text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Evaluate Conversation <ChevronRight size={16} />
                  </button>
                </div>
              </div>
            </div>

            {/* Error Banner */}
            {error && (
              <div className="mt-6 rounded-lg border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-300 flex items-start justify-between gap-3">
                <div className="flex items-start gap-3">
                  <CircleAlert size={18} className="mt-0.5 shrink-0 text-rose-400" />
                  <div>
                    <p className="font-medium text-rose-200">Evaluation Error</p>
                    <p className="mt-1 text-xs text-rose-300/90 leading-relaxed">{error}</p>
                    {error.includes('FastAPI') || error.includes('Model') ? (
                      <p className="mt-2 text-xs font-mono text-amber-300">
                        Tip: Check that Docker or Uvicorn is running on http://localhost:8000 and your Colab GPU server is online.
                      </p>
                    ) : null}
                  </div>
                </div>
                <button onClick={handleEvaluate} className="text-xs font-mono underline hover:text-white">
                  Retry
                </button>
              </div>
            )}

            {/* Presets Demo Area */}
            <div className="mt-10 border-t border-white/10 pt-6">
              <p className="text-xs font-medium uppercase tracking-wider text-slate-400 mb-3">
                Try a Benchmark Example
              </p>
              <div className="grid gap-2 sm:grid-cols-2">
                {DEMO_SNIPPETS.map((snippet, idx) => (
                  <button
                    key={idx}
                    onClick={() => setText(snippet.text)}
                    className="group text-left rounded-lg border border-white/10 bg-[#101318] p-3.5 transition hover:border-cyan-500/30 hover:bg-white/[0.03]"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-cyan-400 group-hover:text-cyan-300">
                        {snippet.label}
                      </span>
                      <span className="rounded bg-white/5 px-2 py-0.5 text-[10px] font-mono text-slate-400 border border-white/10">
                        {snippet.badge}
                      </span>
                    </div>
                    <p className="mt-2 text-xs text-slate-300 line-clamp-2 leading-relaxed italic">
                      "{snippet.text}"
                    </p>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : loading ? (
          /* Step-by-Step Loading View */
          <div className="mx-auto max-w-md pt-24 text-center">
            <div className="mx-auto mb-6 flex size-12 items-center justify-center rounded-full border-2 border-cyan-500 border-t-transparent animate-spin" />
            <h2 className="text-xl font-semibold text-white">Evaluating Conversation</h2>
            <p className="mt-2 text-xs text-slate-400">Processing transcript through multi-stage ML pipeline...</p>

            <div className="mt-8 flex flex-col gap-3 text-left">
              {[
                { step: 1, text: 'Retrieving candidate facets (BM25 + Vector RRF)' },
                { step: 2, text: 'Filtering unobservable facets (Medical & External logs)' },
                { step: 3, text: 'Scoring observable facets via Qwen GPU server' },
                { step: 4, text: 'Validating evidence quotes & abstention invariants' }
              ].map((s) => (
                <div
                  key={s.step}
                  className={`flex items-center gap-3 rounded-lg border p-3 text-xs transition ${
                    loadingStep >= s.step
                      ? 'border-cyan-500/30 bg-cyan-500/10 text-cyan-200'
                      : 'border-white/5 bg-white/[0.02] text-slate-500'
                  }`}
                >
                  <span
                    className={`flex size-5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold ${
                      loadingStep >= s.step ? 'bg-cyan-500 text-slate-950' : 'bg-white/10 text-slate-400'
                    }`}
                  >
                    {s.step}
                  </span>
                  <span>{s.text}</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          /* Evaluation Results View */
          <div>
            {/* Header Controls */}
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-6">
              <div>
                <span className="text-xs font-mono uppercase tracking-widest text-cyan-400 font-semibold">
                  Evaluation Complete
                </span>
                <h1 className="mt-1 text-2xl font-semibold text-white">Facet Evaluation Results</h1>
                <p className="mt-1 text-xs text-slate-400 font-mono">
                  {result?.metadata?.retrieved_count ?? result?.results.length} relevant facets evaluated —{' '}
                  <span className="text-emerald-400 font-medium">{result?.metadata?.scored_count} Scored</span> ·{' '}
                  <span className="text-amber-400 font-medium">{result?.metadata?.abstained_count} Abstained</span>
                </p>
              </div>

              <button
                onClick={() => {
                  setResult(null)
                  setText('')
                }}
                className="rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-xs font-medium text-slate-300 hover:bg-white/10 hover:text-white transition"
              >
                Evaluate Another Transcript
              </button>
            </div>

            {/* Ordinal Scale Legend */}
            <div className="mt-6 rounded-lg border border-white/10 bg-[#101318] p-4 text-xs">
              <div className="flex items-center gap-2 text-slate-300 font-medium mb-2">
                <HelpCircle size={14} className="text-cyan-400" />
                <span>5-Level Ordinal Scoring Scale Key:</span>
              </div>
              <div className="grid grid-cols-5 gap-2 text-center font-mono text-[11px]">
                {[1, 2, 3, 4, 5].map((lvl) => (
                  <div key={lvl} className="rounded bg-white/5 py-1.5 border border-white/5">
                    <span className="font-bold text-cyan-400">{lvl}</span> — {SCALE_LABELS[lvl]}
                  </div>
                ))}
              </div>
            </div>

            {/* Results Grid */}
            <div className="mt-8 space-y-8">
              {/* Section 1: SCORED FACETS */}
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-wider text-emerald-400 flex items-center gap-2 mb-4">
                  <CheckCircle2 size={16} /> Scored Facets ({result?.results.filter((r) => r.status === 'scored').length})
                </h2>

                <div className="grid gap-4">
                  {result?.results
                    .filter((r) => r.status === 'scored')
                    .map((item) => (
                      <FacetCard key={item.facet_id} item={item} onSelect={() => setSelectedFacet(item)} />
                    ))}
                </div>
              </div>

              {/* Section 2: ABSTAINED FACETS */}
              {result?.results.some((r) => r.status !== 'scored') && (
                <div className="pt-4 border-t border-white/10">
                  <h2 className="text-sm font-semibold uppercase tracking-wider text-amber-400 flex items-center gap-2 mb-4">
                    <ShieldAlert size={16} /> Abstained Facets ({result?.results.filter((r) => r.status !== 'scored').length})
                  </h2>
                  <p className="text-xs text-slate-400 mb-4">
                    Abstention is intentional: the model explicitly avoids scoring when constructs are unobservable or conversational evidence is insufficient.
                  </p>

                  <div className="grid gap-4 sm:grid-cols-2">
                    {result?.results
                      .filter((r) => r.status !== 'scored')
                      .map((item) => (
                        <FacetCard key={item.facet_id} item={item} onSelect={() => setSelectedFacet(item)} />
                      ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Side Evidence Drawer */}
      {selectedFacet && (
        <EvidenceDrawer facet={selectedFacet} onClose={() => setSelectedFacet(null)} />
      )}
    </div>
  )
}

// Facet Card Component
function FacetCard({ item, onSelect }: { item: FacetScoreResult; onSelect: () => void }) {
  const isScored = item.status === 'scored'
  const isNotObservable = item.status === 'not_observable'
  const isInferenceError = item.status === 'inference_error'

  return (
    <div
      onClick={onSelect}
      className={`group cursor-pointer rounded-xl border p-5 transition ${
        isScored
          ? 'border-white/10 bg-[#101318] hover:border-cyan-500/40 hover:bg-white/[0.02]'
          : isInferenceError
          ? 'border-rose-500/30 bg-rose-500/[0.03] hover:border-rose-500/50'
          : 'border-amber-500/20 bg-amber-500/[0.02] hover:border-amber-500/40'
      }`}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-white group-hover:text-cyan-300 transition">
              {item.facet}
            </h3>
            <span className="font-mono text-[10px] text-slate-500">[{item.facet_id}]</span>
          </div>
        </div>

        {/* Status Badge & Score */}
        <div className="flex items-center gap-3">
          {isScored ? (
            <div className="flex items-center gap-2">
              <span className="rounded bg-emerald-500/10 px-2.5 py-1 text-xs font-mono text-emerald-400 border border-emerald-500/20 font-bold">
                Score: {item.score}/5 ({SCALE_LABELS[item.score ?? 3]})
              </span>
              <span className="text-xs font-mono text-slate-400">
                Confidence: {Math.round((item.confidence ?? 0.85) * 100)}%
              </span>
            </div>
          ) : (
            <span
              className={`rounded px-2.5 py-1 text-xs font-mono font-semibold border ${
                isInferenceError
                  ? 'bg-rose-500/10 text-rose-300 border-rose-500/30'
                  : isNotObservable
                  ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                  : 'bg-slate-500/10 text-slate-300 border-slate-500/30'
              }`}
            >
              {isInferenceError
                ? 'INFERENCE ERROR'
                : isNotObservable
                ? 'NOT OBSERVABLE'
                : 'INSUFFICIENT EVIDENCE'}
            </span>
          )}
        </div>
      </div>

      {/* Prominent Evidence Box */}
      {isScored && item.evidence && (
        <div className="mt-4 rounded-lg border border-cyan-500/20 bg-cyan-500/5 p-3.5 text-xs">
          <div className="flex items-center gap-2 font-mono text-[11px] text-cyan-400 font-semibold mb-1">
            <Quote size={13} /> Extracted Conversational Evidence:
          </div>
          <p className="text-slate-200 italic font-serif leading-relaxed">
            "{item.evidence}"
          </p>
        </div>
      )}

      {/* Reasoning */}
      <p className="mt-3 text-xs text-slate-400 leading-relaxed">
        <span className="font-semibold text-slate-300">Reason:</span> {item.reason}
      </p>
    </div>
  )
}

// Side Evidence Drawer Component
function EvidenceDrawer({ facet, onClose }: { facet: FacetScoreResult; onClose: () => void }) {
  const isScored = facet.status === 'scored'

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <aside
        onClick={(e) => e.stopPropagation()}
        className="absolute inset-y-0 right-0 w-full max-w-lg border-l border-white/10 bg-[#0d1016] p-6 shadow-2xl overflow-y-auto"
      >
        <div className="flex items-center justify-between border-b border-white/10 pb-4">
          <span className="text-xs font-mono uppercase tracking-widest text-cyan-400 font-semibold">
            Facet Evidence Inspection
          </span>
          <button
            onClick={onClose}
            className="rounded-lg p-2 text-slate-400 hover:bg-white/10 hover:text-white transition"
          >
            <X size={18} />
          </button>
        </div>

        <div className="mt-6">
          <h2 className="text-2xl font-bold text-white">{facet.facet}</h2>
          <p className="font-mono text-xs text-slate-500 mt-1">Facet ID: {facet.facet_id}</p>
        </div>

        <div className="mt-6 flex items-center gap-4 border-y border-white/10 py-4">
          {isScored ? (
            <>
              <div>
                <p className="text-[10px] font-mono text-slate-400 uppercase">Assigned Score</p>
                <p className="text-xl font-bold text-emerald-400 font-mono">
                  {facet.score} / 5 <span className="text-xs text-slate-300">({SCALE_LABELS[facet.score ?? 3]})</span>
                </p>
              </div>
              <div className="border-l border-white/10 pl-4">
                <p className="text-[10px] font-mono text-slate-400 uppercase">Model Confidence</p>
                <p className="text-xl font-bold text-cyan-400 font-mono">
                  {Math.round((facet.confidence ?? 0.85) * 100)}%
                </p>
              </div>
            </>
          ) : (
            <div>
              <p className="text-[10px] font-mono text-slate-400 uppercase">Abstention Status</p>
              <p className="text-sm font-bold text-amber-400 font-mono uppercase mt-0.5">
                {facet.status === 'not_observable' ? 'Not Observable Construct' : 'Insufficient Evidence'}
              </p>
            </div>
          )}
        </div>

        {/* Evidence Detail */}
        <div className="mt-6 space-y-6">
          {isScored && facet.evidence ? (
            <div>
              <p className="text-xs font-mono uppercase text-slate-400 font-semibold mb-2">
                Quoted Conversational Evidence
              </p>
              <div className="rounded-lg border border-cyan-500/30 bg-cyan-500/10 p-4 text-xs text-slate-200 leading-relaxed italic">
                "{facet.evidence}"
              </div>
            </div>
          ) : null}

          <div>
            <p className="text-xs font-mono uppercase text-slate-400 font-semibold mb-2">
              Model Reasoning
            </p>
            <p className="text-xs text-slate-300 leading-relaxed rounded-lg border border-white/10 bg-white/[0.02] p-4">
              {facet.reason}
            </p>
          </div>

          {!isScored && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 text-xs text-amber-200">
              <p className="font-semibold flex items-center gap-1.5 mb-1">
                <ShieldAlert size={15} /> Abstention Design Guarantee
              </p>
              <p className="text-[11px] text-amber-200/90 leading-relaxed">
                The system intentionally assigns <code className="font-mono text-amber-300">score = null</code> to prevent false medical diagnostic or ungrounded biographical hallucinations.
              </p>
            </div>
          )}

          {/* Scale Legend Reference */}
          <div className="border-t border-white/10 pt-4 text-xs">
            <p className="font-mono text-[11px] text-slate-400 mb-2">5-Level Ordinal Scale Reference:</p>
            <div className="space-y-1 font-mono text-[11px] text-slate-300">
              <p>1 — Very Low (Minimal or negative indication)</p>
              <p>2 — Low (Slight indication)</p>
              <p>3 — Moderate (Noticeable indication)</p>
              <p>4 — High (Strong clear evidence)</p>
              <p>5 — Very High (Exceptional primary characteristic)</p>
            </div>
          </div>
        </div>
      </aside>
    </div>
  )
}
