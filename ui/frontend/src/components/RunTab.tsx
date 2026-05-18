import { useEffect, useRef, useState } from 'react'
import type { Results } from '../types'
import { ResultsDisplay } from './ResultsDisplay'
import { Play, Loader2, CheckCircle2, XCircle, Terminal, BarChart3, Square, Key } from 'lucide-react'

type RunState = 'idle' | 'running' | 'done' | 'error' | 'interrupted'
type Provider = 'ollama' | 'openai'

const PHASES = [
  { id: 0, label: 'Scope' },
  { id: 1, label: 'Extract' },
  { id: 2, label: 'Retrieve' },
  { id: 3, label: 'Classify' },
  { id: 4, label: 'Evaluate' },
]

function detectPhase(lines: string[]): number {
  let phase = -1
  for (const l of lines) {
    if (l.includes('PHASE 0') || l.includes('Phase 0') || l.includes('scope detection')) phase = 0
    else if (l.includes('PHASE 1') || l.includes('Phase 1') || l.includes('extraction')) phase = 1
    else if (l.includes('PHASE 2') || l.includes('Phase 2') || l.includes('retrieval')) phase = 2
    else if (l.includes('PHASE 3') || l.includes('Phase 3') || l.includes('Classification') || l.includes('classify')) phase = 3
    else if (l.includes('PHASE 4') || l.includes('Phase 4') || l.includes('Evaluation') || l.includes('evaluate')) phase = 4
  }
  return phase
}

function colorLine(line: string): string {
  if (line.includes('━━━') || line.includes('PHASE') || line.includes('Phase')) return '#7dd3fc'
  if (line.includes('✓') || line.includes('Done') || line.includes('PASSED') || line.includes('completed')) return '#34d399'
  if (line.includes('ERROR') || line.includes('error') || line.includes('FAIL')) return '#f87171'
  if (line.includes('SKIP') || line.includes('warn') || line.includes('already done')) return '#fbbf24'
  if (line.includes('[') && line.includes(']')) return '#cbd5e1'
  return '#8b949e'
}

function OptionToggle({
  checked, onChange, label, description, color,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label: string
  description: string
  color: string
}) {
  return (
    <button
      onClick={() => onChange(!checked)}
      title={description}
      style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '7px 12px', borderRadius: 8, cursor: 'pointer',
        border: checked ? `1.5px solid ${color}` : '1.5px solid #e2e8f0',
        background: checked ? `${color}12` : '#fff',
        fontSize: 13, fontWeight: 500,
        color: checked ? color : '#475569',
        transition: 'all 150ms ease',
        userSelect: 'none',
      }}
    >
      <span style={{
        width: 14, height: 14, borderRadius: 3, flexShrink: 0,
        border: checked ? `2px solid ${color}` : '2px solid #cbd5e1',
        background: checked ? color : 'transparent',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        transition: 'all 150ms ease',
      }}>
        {checked && <span style={{ color: '#fff', fontSize: 9, lineHeight: 1 }}>✓</span>}
      </span>
      {label}
    </button>
  )
}

function ProviderChip({
  value, current, onChange, label,
}: {
  value: Provider
  current: Provider
  onChange: (v: Provider) => void
  label: string
}) {
  const active = value === current
  return (
    <button
      onClick={() => onChange(value)}
      style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '7px 12px', borderRadius: 8, cursor: 'pointer',
        border: active ? '1.5px solid #1d4ed8' : '1.5px solid #e2e8f0',
        background: active ? '#eff6ff' : '#fff',
        fontSize: 13, fontWeight: 500,
        color: active ? '#1d4ed8' : '#475569',
        transition: 'all 150ms ease',
        userSelect: 'none',
      }}
    >
      <span style={{
        width: 14, height: 14, borderRadius: '50%', flexShrink: 0,
        border: active ? '2px solid #1d4ed8' : '2px solid #cbd5e1',
        background: active ? '#1d4ed8' : 'transparent',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        transition: 'all 150ms ease',
      }}>
        {active && <span style={{ width: 5, height: 5, borderRadius: '50%', background: '#fff', display: 'block' }} />}
      </span>
      {label}
    </button>
  )
}

export function RunTab() {
  const [runState, setRunState] = useState<RunState>('idle')
  const [lines, setLines] = useState<string[]>([])
  const [results, setResults] = useState<Results | null>(null)
  const termRef = useRef<HTMLDivElement>(null)
  const esRef = useRef<EventSource | null>(null)

  // Run options
  const [smokeTest, setSmokeTest] = useState(false)
  const [force, setForce] = useState(false)
  const [skipScope, setSkipScope] = useState(false)
  const [provider, setProvider] = useState<Provider>('ollama')
  const [apiKey, setApiKey] = useState('')
  const [hasEnvKey, setHasEnvKey] = useState(false)

  useEffect(() => {
    fetch('/api/run/status')
      .then(r => r.json())
      .then(s => { if (s.running) { setRunState('running'); attach() } })
      .catch(console.error)
    fetch('/api/config')
      .then(r => r.json())
      .then(c => {
        setProvider(c.provider as Provider)
        setHasEnvKey(c.has_api_key)
      })
      .catch(console.error)
  }, [])

  useEffect(() => {
    if (termRef.current) termRef.current.scrollTop = termRef.current.scrollHeight
  }, [lines])

  function attach() {
    esRef.current?.close()
    const es = new EventSource('/api/run/stream')
    esRef.current = es
    setLines([])
    setResults(null)

    es.onmessage = e => {
      const data: string = e.data
      if (data === '__DONE__') {
        es.close()
        fetch('/api/results')
          .then(r => r.json())
          .then(r => { setResults(r); setRunState('done') })
          .catch(() => setRunState('done'))
      } else {
        setLines(prev => [...prev, data])
      }
    }

    es.onerror = () => {
      es.close()
      setRunState(prev => prev === 'running' ? 'error' : prev)
    }
  }

  async function startRun() {
    setRunState('running')
    setLines([])
    setResults(null)
    try {
      const body = {
        smoke_test: smokeTest,
        force,
        skip_scope: skipScope,
        provider,
        api_key: provider === 'openai' && !hasEnvKey ? apiKey : '',
      }
      const res = await fetch('/api/run/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const err = await res.json()
        setLines([`Error: ${err.detail}`])
        setRunState('error')
        return
      }
      attach()
    } catch (e) {
      setLines([`Failed to reach backend: ${e}`])
      setRunState('error')
    }
  }

  async function stopRun() {
    try {
      await fetch('/api/run/stop', { method: 'POST' })
      esRef.current?.close()
      setRunState('interrupted')
    } catch (e) {
      console.error('Failed to stop run:', e)
    }
  }

  const currentPhase = runState === 'running' ? detectPhase(lines) : -1
  const isRunning = runState === 'running'
  const isStopped = runState === 'interrupted'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* ── Control card ──────────────────────────────────────────────────── */}
      <div
        style={{
          background: '#ffffff',
          borderRadius: 16,
          overflow: 'hidden',
          border: '1px solid #e2e8f0',
          boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.05), 0 1px 2px -1px rgb(0 0 0 / 0.04)',
        }}
      >
        {/* Top section */}
        <div style={{ padding: '28px 28px 24px 28px' }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 24 }}>

            <div style={{ flex: 1 }}>
              <p style={{
                fontSize: 11, fontWeight: 600, color: '#94a3b8',
                textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6,
              }}>
                Detection Pipeline
              </p>
              <h2 style={{ fontSize: 20, fontWeight: 600, color: '#0f172a', marginBottom: 8, lineHeight: 1.3 }}>
                Run full compliance check
              </h2>
              <p style={{ fontSize: 14, color: '#64748b', lineHeight: 1.65, maxWidth: 560 }}>
                Runs all 5 phases across Hetzner, Zalando, and Trade Republic. Results are saved
                and appear in the History tab after completion.
              </p>
            </div>

            <div style={{ display: 'flex', gap: 8, flexShrink: 0, alignItems: 'center' }}>
              {isRunning && (
                <button
                  onClick={stopRun}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    padding: '10px 20px', borderRadius: 10,
                    fontSize: 13, fontWeight: 600,
                    background: '#fff1f2', color: '#be123c',
                    border: '1px solid #fecdd3', cursor: 'pointer',
                    transition: 'background 150ms ease',
                  }}
                  onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = '#ffe4e6' }}
                  onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = '#fff1f2' }}
                >
                  <Square size={13} fill="currentColor" />
                  Interrupt
                </button>
              )}
              <button
                onClick={startRun}
                disabled={isRunning}
                style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  padding: '10px 22px', borderRadius: 10,
                  fontSize: 13, fontWeight: 600, color: '#fff',
                  background: isRunning ? '#93c5fd' : 'linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%)',
                  boxShadow: isRunning ? 'none' : '0 2px 8px rgb(29 78 216 / 0.28)',
                  cursor: isRunning ? 'not-allowed' : 'pointer',
                  opacity: isRunning ? 0.7 : 1,
                  transition: 'all 150ms ease', border: 'none',
                }}
              >
                {isRunning
                  ? <><Loader2 size={14} className="animate-spin" />Running…</>
                  : <><Play size={14} fill="currentColor" />Start Pipeline</>
                }
              </button>
            </div>
          </div>

          {/* ── Run options ──────────────────────────────────────────────── */}
          <div style={{
            marginTop: 20,
            padding: '16px 18px',
            background: '#f8fafc',
            borderRadius: 12,
            border: '1px solid #e2e8f0',
            display: 'flex', flexDirection: 'column', gap: 14,
            opacity: isRunning ? 0.5 : 1,
            pointerEvents: isRunning ? 'none' : 'auto',
            transition: 'opacity 150ms ease',
          }}>
            {/* Row 1: flags */}
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              <OptionToggle
                checked={smokeTest}
                onChange={v => { setSmokeTest(v); if (v) setForce(false) }}
                label="Smoke test"
                description="Quick slice of Hetzner only — verifies the full chain without running all use cases"
                color="#7c3aed"
              />
              <OptionToggle
                checked={force}
                onChange={v => { setForce(v); if (v) setSmokeTest(false); if (!v) setSkipScope(false) }}
                label="Force re-run"
                description="Ignore skip guards and re-run all phases from scratch"
                color="#0369a1"
              />
              <div style={{ opacity: smokeTest ? 0.4 : 1, pointerEvents: smokeTest ? 'none' : 'auto' }}>
                <OptionToggle
                  checked={skipScope}
                  onChange={v => { setSkipScope(v); if (v) setForce(true) }}
                  label="Skip scope (Phase 0)"
                  description="Skip original-policy scope detection and force re-run phases 1–5"
                  color="#b45309"
                />
              </div>
            </div>

            {/* Row 2: provider */}
            <div>
              <p style={{ fontSize: 11, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
                LLM Provider
              </p>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                <ProviderChip value="ollama" current={provider} onChange={setProvider} label="Local (Ollama)" />
                <ProviderChip value="openai" current={provider} onChange={setProvider} label="API (OpenAI)" />

                {provider === 'openai' && !hasEnvKey && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginLeft: 4 }}>
                    <Key size={13} style={{ color: '#64748b', flexShrink: 0 }} />
                    <input
                      type="password"
                      placeholder="sk-…  OpenAI API key"
                      value={apiKey}
                      onChange={e => setApiKey(e.target.value)}
                      style={{
                        fontSize: 13, padding: '6px 10px',
                        borderRadius: 8, border: '1px solid #cbd5e1',
                        background: '#fff', color: '#0f172a',
                        outline: 'none', width: 240,
                        fontFamily: 'ui-monospace, monospace',
                      }}
                    />
                  </div>
                )}
                {provider === 'openai' && hasEnvKey && (
                  <span style={{ fontSize: 12, color: '#16a34a', display: 'flex', alignItems: 'center', gap: 4, marginLeft: 4 }}>
                    <Key size={12} /> API key from environment
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Phase progress strip */}
        <div style={{
          display: 'flex', alignItems: 'center',
          padding: '14px 28px',
          background: '#f8fafc', borderTop: '1px solid #f1f5f9',
        }}>
          {PHASES.map((p, i) => {
            const done   = runState === 'done' || (isRunning && currentPhase > p.id)
            const active = isRunning && currentPhase === p.id

            return (
              <div key={p.id} style={{ display: 'flex', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{
                    width: 24, height: 24, borderRadius: '50%',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 11, fontWeight: 700,
                    background: done ? '#22c55e' : active ? '#1d4ed8' : '#e2e8f0',
                    color: done || active ? '#fff' : '#94a3b8',
                    boxShadow: active ? '0 0 0 3px rgb(29 78 216 / 0.2)' : 'none',
                    transition: 'all 300ms',
                  }}>
                    {done ? '✓' : p.id}
                  </div>
                  <span style={{
                    fontSize: 12, fontWeight: 500,
                    color: done ? '#16a34a' : active ? '#1d4ed8' : '#94a3b8',
                    transition: 'color 300ms',
                  }}>
                    {p.label}
                  </span>
                </div>
                {i < PHASES.length - 1 && (
                  <div style={{
                    margin: '0 12px', height: 1, width: 32,
                    background: done ? '#86efac' : '#e2e8f0',
                    transition: 'background 500ms',
                  }} />
                )}
              </div>
            )
          })}

          {runState === 'done' && (
            <span style={{
              marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6,
              fontSize: 12, fontWeight: 600, color: '#16a34a',
            }}>
              <CheckCircle2 size={14} /> Complete
            </span>
          )}
          {runState === 'idle' && (
            <span style={{ marginLeft: 'auto', fontSize: 12, color: '#94a3b8' }}>
              Ready to run
            </span>
          )}
        </div>

        {/* Status banners */}
        {runState === 'done' && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            margin: '0 28px 20px 28px', padding: '12px 16px',
            borderRadius: 12, fontSize: 14, fontWeight: 500,
            background: '#f0fdf4', color: '#15803d', border: '1px solid #bbf7d0',
          }}>
            <CheckCircle2 size={16} />
            Pipeline completed — results saved and displayed below.
          </div>
        )}
        {runState === 'error' && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            margin: '0 28px 20px 28px', padding: '12px 16px',
            borderRadius: 12, fontSize: 14, fontWeight: 500,
            background: '#fef2f2', color: '#b91c1c', border: '1px solid #fecaca',
          }}>
            <XCircle size={16} />
            Pipeline exited with an error — see log output below.
          </div>
        )}
        {isStopped && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            margin: '0 28px 20px 28px', padding: '12px 16px',
            borderRadius: 12, fontSize: 14, fontWeight: 500,
            background: '#fffbeb', color: '#92400e', border: '1px solid #fde68a',
          }}>
            <Square size={16} fill="currentColor" />
            Pipeline was interrupted. You can start a new run at any time.
          </div>
        )}
      </div>

      {/* ── Live terminal ─────────────────────────────────────────────────── */}
      {lines.length > 0 && (
        <div style={{
          background: '#ffffff', borderRadius: 16, overflow: 'hidden',
          border: '1px solid #e2e8f0', boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.05)',
        }}>
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '12px 20px',
            borderBottom: '1px solid #f1f5f9', background: '#f8fafc',
          }}>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              fontSize: 11, fontWeight: 600, color: '#64748b',
              textTransform: 'uppercase', letterSpacing: '0.08em',
            }}>
              <Terminal size={13} />
              Live Output
            </div>
            {isRunning ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 500, color: '#2563eb' }}>
                <span className="pulse-dot" style={{ width: 6, height: 6, borderRadius: '50%', background: '#2563eb', display: 'inline-block' }} />
                Streaming
              </div>
            ) : (
              <span style={{ fontSize: 12, color: '#94a3b8' }}>{lines.length} lines</span>
            )}
          </div>
          <div
            className="log-terminal"
            style={{ height: 420, borderRadius: 0 }}
            ref={termRef}
          >
            {lines.map((line, i) => (
              <div key={i} style={{ color: colorLine(line) }}>{line || ' '}</div>
            ))}
          </div>
        </div>
      )}

      {/* ── Results ───────────────────────────────────────────────────────── */}
      {results && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
            <BarChart3 size={18} style={{ color: '#94a3b8' }} />
            <h2 style={{ fontSize: 15, fontWeight: 600, color: '#1e293b' }}>Evaluation Results</h2>
          </div>
          <ResultsDisplay results={results} />
        </div>
      )}
    </div>
  )
}
