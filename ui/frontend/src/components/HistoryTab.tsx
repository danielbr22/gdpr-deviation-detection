import { useEffect, useState } from 'react'
import type { LogEntry, Results } from '../types'
import { formatTs } from '../lib/utils'
import { ResultsDisplay } from './ResultsDisplay'
import { ChevronDown, Clock, FileCode2, CheckCircle2, MousePointerClick, Terminal, BarChart3 } from 'lucide-react'

export function HistoryTab() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [logContent, setLogContent] = useState<string>('')
  const [results, setResults] = useState<Results | null>(null)
  const [logExpanded, setLogExpanded] = useState(false)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch('/api/logs')
      .then(r => r.json())
      .then(setLogs)
      .catch(console.error)
  }, [])

  async function select(filename: string) {
    if (selected === filename) return
    setSelected(filename)
    setResults(null)
    setLogContent('')
    setLogExpanded(false)
    setLoading(true)

    try {
      const [logRes, resRes] = await Promise.allSettled([
        fetch(`/api/logs/${filename}/content`).then(r => r.json()),
        fetch(`/api/logs/${filename}/results`).then(r => r.ok ? r.json() : null),
      ])
      if (logRes.status === 'fulfilled') setLogContent(logRes.value.content ?? '')
      if (resRes.status === 'fulfilled' && resRes.value) setResults(resRes.value)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', gap: 20, minHeight: 600 }}>

      {/* ── Left sidebar ──────────────────────────────────────────────────── */}
      <aside style={{
        width: 288, flexShrink: 0,
        background: '#ffffff', borderRadius: 16,
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
        border: '1px solid #e2e8f0',
        boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.05)',
      }}>
        {/* Sidebar header */}
        <div style={{
          padding: '14px 20px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          borderBottom: '1px solid #f1f5f9', background: '#f8fafc', flexShrink: 0,
        }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            fontSize: 11, fontWeight: 600, color: '#64748b',
            textTransform: 'uppercase', letterSpacing: '0.08em',
          }}>
            <Clock size={12} />
            Run History
          </div>
          <span style={{
            fontSize: 11, fontWeight: 600, padding: '2px 8px',
            borderRadius: 999, background: '#eff6ff', color: '#1d4ed8', border: '1px solid #bfdbfe',
          }}>
            {logs.length}
          </span>
        </div>

        {/* Run list */}
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {logs.length === 0 && (
            <div style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              justifyContent: 'center', gap: 12, padding: '48px 24px', textAlign: 'center',
            }}>
              <FileCode2 size={32} style={{ color: '#e2e8f0' }} />
              <p style={{ fontSize: 13, color: '#94a3b8' }}>
                No log files found in <code style={{ fontSize: 11 }}>logs/</code>
              </p>
            </div>
          )}

          {logs.map(log => {
            const isSelected = selected === log.filename
            return (
              <button
                key={log.filename}
                onClick={() => select(log.filename)}
                style={{
                  width: '100%', textAlign: 'left', cursor: 'pointer', outline: 'none',
                  display: 'flex', alignItems: 'flex-start', gap: 12,
                  padding: '14px 20px',
                  borderBottom: '1px solid #f8fafc',
                  background: isSelected ? '#eff6ff' : 'transparent',
                  borderLeft: isSelected ? '3px solid #2563eb' : '3px solid transparent',
                  transition: 'background 150ms ease',
                  border: 'none',
                  borderBottomWidth: 1, borderBottomStyle: 'solid', borderBottomColor: '#f8fafc',
                  borderLeftWidth: 3, borderLeftStyle: 'solid',
                  borderLeftColor: isSelected ? '#2563eb' : 'transparent',
                }}
              >
                <div style={{
                  width: 32, height: 32, borderRadius: 8, flexShrink: 0, marginTop: 1,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  background: isSelected ? '#dbeafe' : '#f1f5f9',
                }}>
                  <FileCode2 size={15} style={{ color: isSelected ? '#2563eb' : '#94a3b8' }} />
                </div>

                <div style={{ minWidth: 0, flex: 1 }}>
                  <p style={{
                    fontSize: 12, fontWeight: 600, lineHeight: 1.3,
                    color: isSelected ? '#1e40af' : '#334155',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {formatTs(log.timestamp)}
                  </p>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 5 }}>
                    <span style={{ fontSize: 12, color: '#94a3b8' }}>
                      {(log.size_bytes / 1024).toFixed(0)} KB
                    </span>
                    {log.has_results && (
                      <span style={{
                        display: 'inline-flex', alignItems: 'center', gap: 4,
                        fontSize: 11, fontWeight: 500, padding: '2px 6px', borderRadius: 4,
                        background: '#f0fdf4', color: '#16a34a',
                      }}>
                        <CheckCircle2 size={9} /> saved
                      </span>
                    )}
                  </div>
                </div>
              </button>
            )
          })}
        </div>
      </aside>

      {/* ── Right detail panel ────────────────────────────────────────────── */}
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 20 }}>

        {/* Empty state */}
        {!selected && (
          <div style={{
            flex: 1, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            gap: 12, textAlign: 'center', padding: '80px 0',
          }}>
            <div style={{
              width: 56, height: 56, borderRadius: 16,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: '#f1f5f9', border: '1px solid #e2e8f0',
            }}>
              <MousePointerClick size={24} style={{ color: '#cbd5e1' }} />
            </div>
            <div>
              <p style={{ fontSize: 14, fontWeight: 500, color: '#64748b' }}>
                Select a run from the sidebar
              </p>
              <p style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>
                Runs with saved results will show full evaluation data
              </p>
            </div>
          </div>
        )}

        {/* Loading */}
        {selected && loading && (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 14, color: '#94a3b8' }}>
              <div
                className="animate-spin"
                style={{ width: 16, height: 16, borderRadius: '50%', border: '2px solid #cbd5e1', borderTopColor: 'transparent' }}
              />
              Loading…
            </div>
          </div>
        )}

        {/* Detail content */}
        {selected && !loading && (
          <>
            {/* No results notice */}
            {!results && (
              <div style={{
                display: 'flex', alignItems: 'flex-start', gap: 12,
                padding: '16px 20px', borderRadius: 16, fontSize: 14,
                background: '#fffbeb', border: '1px solid #fde68a', color: '#92400e',
              }}>
                <span style={{ fontSize: 16, lineHeight: 1 }}>⚠</span>
                <span>No results snapshot for this run. Results are only saved for runs triggered via this UI.</span>
              </div>
            )}

            {/* Results display */}
            {results && (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
                  <BarChart3 size={18} style={{ color: '#94a3b8' }} />
                  <h2 style={{ fontSize: 15, fontWeight: 600, color: '#1e293b' }}>Evaluation Results</h2>
                </div>
                <ResultsDisplay results={results} />
              </div>
            )}

            {/* Raw log toggle */}
            <div style={{
              background: '#ffffff', borderRadius: 16, overflow: 'hidden',
              border: '1px solid #e2e8f0', boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.04)',
            }}>
              <button
                onClick={() => setLogExpanded(e => !e)}
                style={{
                  width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '14px 20px', cursor: 'pointer', outline: 'none', background: 'transparent',
                  border: 'none', transition: 'background 150ms ease',
                }}
                onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = '#f8fafc' }}
                onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = 'transparent' }}
              >
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  fontSize: 11, fontWeight: 600, color: '#64748b',
                  textTransform: 'uppercase', letterSpacing: '0.08em',
                }}>
                  <Terminal size={13} />
                  Raw Log Output
                  {logContent && (
                    <span style={{ marginLeft: 4, color: '#cbd5e1', fontWeight: 400, fontSize: 11, textTransform: 'none', letterSpacing: 'normal' }}>
                      — {logContent.split('\n').length} lines
                    </span>
                  )}
                </div>
                <ChevronDown
                  size={16}
                  style={{
                    color: '#cbd5e1',
                    transform: logExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 200ms ease',
                  }}
                />
              </button>
              {logExpanded && logContent && (
                <div
                  className="log-terminal"
                  style={{ borderRadius: 0, maxHeight: 480, borderTop: '1px solid #21262d' }}
                >
                  {logContent}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
