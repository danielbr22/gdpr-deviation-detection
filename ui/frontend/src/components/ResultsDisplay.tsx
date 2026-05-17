import type { Results, UseCaseResult } from '../types'
import { pct } from '../lib/utils'

/* ── Deviation type colours ──────────────────────────────────────────────── */
const TYPE_STYLE: Record<string, { bg: string; text: string; dot: string }> = {
  constraint_coverage: { bg: '#f3e8ff', text: '#7e22ce', dot: '#a855f7' },
  severity:            { bg: '#fff7ed', text: '#c2410c', dot: '#f97316' },
  execution_style:     { bg: '#eff6ff', text: '#1d4ed8', dot: '#3b82f6' },
  negation:            { bg: '#fff1f2', text: '#be123c', dot: '#f43f5e' },
  responsibility:      { bg: '#fefce8', text: '#854d0e', dot: '#eab308' },
  data:                { bg: '#f0fdfa', text: '#0f766e', dot: '#14b8a6' },
  missing_coverage:    { bg: '#f8fafc', text: '#475569', dot: '#94a3b8' },
}

function Badge({ type }: { type: string }) {
  const s = TYPE_STYLE[type] ?? TYPE_STYLE.missing_coverage
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '2px 8px', borderRadius: 4,
      fontSize: 12, fontWeight: 500,
      background: s.bg, color: s.text,
    }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', flexShrink: 0, background: s.dot, display: 'inline-block' }} />
      {type.replace(/_/g, ' ')}
    </span>
  )
}

function scoreColor(v: number) {
  if (v >= 0.7) return '#16a34a'
  if (v >= 0.4) return '#d97706'
  return '#dc2626'
}

function scoreBg(v: number) {
  if (v >= 0.7) return '#dcfce7'
  if (v >= 0.4) return '#fef3c7'
  return '#fee2e2'
}

function MiniBar({ value, color }: { value: number; color: string }) {
  return (
    <div style={{ marginTop: 8, height: 4, borderRadius: 2, overflow: 'hidden', background: '#f1f5f9' }}>
      <div style={{
        height: '100%', borderRadius: 2,
        width: `${Math.max(2, value * 100)}%`,
        background: color,
        transition: 'width 500ms ease',
      }} />
    </div>
  )
}

function KpiCard({ label, value, sub }: { label: string; value: number; sub?: string }) {
  const c = scoreColor(value)
  const bg = scoreBg(value)
  return (
    <div style={{
      background: '#ffffff', borderRadius: 16,
      padding: 20, display: 'flex', flexDirection: 'column', gap: 2,
      border: '1px solid #e2e8f0', boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.04)',
    }}>
      <span style={{ fontSize: 11, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
        {label}
      </span>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8, marginTop: 4 }}>
        <span style={{ fontSize: 30, fontWeight: 700, lineHeight: 1, color: c }}>{pct(value)}</span>
        {sub && <span style={{ fontSize: 12, color: '#94a3b8', marginBottom: 2 }}>{sub}</span>}
      </div>
      <MiniBar value={value} color={c} />
      <div style={{
        marginTop: 8, alignSelf: 'flex-start',
        padding: '2px 8px', borderRadius: 4,
        fontSize: 12, fontWeight: 500,
        background: bg, color: c,
      }}>
        {value >= 0.7 ? 'Good' : value >= 0.4 ? 'Moderate' : 'Low'}
      </div>
    </div>
  )
}

function DetectedPill({ detected }: { detected: boolean }) {
  return detected ? (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      fontSize: 12, fontWeight: 600, padding: '2px 8px', borderRadius: 999,
      background: '#dcfce7', color: '#15803d',
    }}>
      <span>✓</span> Yes
    </span>
  ) : (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      fontSize: 12, fontWeight: 600, padding: '2px 8px', borderRadius: 999,
      background: '#fee2e2', color: '#b91c1c',
    }}>
      <span>✗</span> No
    </span>
  )
}

function UseCaseSection({ uc }: { uc: UseCaseResult }) {
  const { overall, type_metrics, gold_results } = uc
  const detected = gold_results.filter(g => g.detected).length
  const total    = gold_results.length
  const pctDet   = total > 0 ? detected / total : 0

  const COMPANY: Record<string, string> = {
    hetzner:       'Hetzner Online GmbH',
    zalando:       'Zalando SE',
    traderepublic: 'Trade Republic Bank GmbH',
  }

  return (
    <div style={{
      background: '#ffffff', borderRadius: 16, overflow: 'hidden', marginBottom: 24,
      border: '1px solid #e2e8f0', boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.05)',
    }}>
      {/* Section header */}
      <div style={{
        padding: '20px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        borderBottom: '1px solid #f1f5f9', background: '#f8fafc',
      }}>
        <div>
          <h3 style={{ fontSize: 14, fontWeight: 700, color: '#1e293b' }}>
            {COMPANY[uc.use_case] ?? uc.use_case}
          </h3>
          <p style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>
            {detected} of {total} gold-standard deviations detected
            <span style={{ marginLeft: 8, fontWeight: 600, color: scoreColor(pctDet) }}>
              ({pct(pctDet)})
            </span>
          </p>
        </div>
        <div style={{
          padding: '8px 16px', borderRadius: 12, textAlign: 'center',
          background: scoreBg(overall.f1),
          border: `1px solid ${scoreColor(overall.f1)}30`,
        }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: scoreColor(overall.f1) }}>Overall F1</div>
          <div style={{ fontSize: 22, fontWeight: 700, lineHeight: 1, marginTop: 2, color: scoreColor(overall.f1) }}>
            {pct(overall.f1)}
          </div>
        </div>
      </div>

      <div style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 24 }}>

        {/* KPI row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
          <KpiCard label="Precision" value={overall.precision} sub={`${overall.tp}TP / ${overall.fp}FP`} />
          <KpiCard label="Recall"    value={overall.recall}    sub={`${overall.tp}TP / ${overall.fn}FN`} />
          <KpiCard label="F1 Score"  value={overall.f1} />
        </div>

        {/* Per-type breakdown */}
        <div>
          <p style={{ fontSize: 11, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>
            Breakdown by deviation type
          </p>
          <div style={{ borderRadius: 12, overflow: 'hidden', border: '1px solid #f1f5f9' }}>
            <table style={{ width: '100%', fontSize: 14, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: '#f8fafc', borderBottom: '1px solid #f1f5f9' }}>
                  {['Type', 'TP', 'FP', 'FN', 'P', 'R', 'F1'].map((h, i) => (
                    <th key={h} style={{
                      padding: '10px 16px', textAlign: i === 0 ? 'left' : 'center',
                      fontSize: 11, fontWeight: 600, color: '#64748b',
                      textTransform: 'uppercase', letterSpacing: '0.06em',
                    }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(type_metrics).map(([type, m], idx, arr) => (
                  <tr key={type} style={{ borderBottom: idx < arr.length - 1 ? '1px solid #f8fafc' : 'none' }}>
                    <td style={{ padding: '10px 16px' }}><Badge type={type} /></td>
                    <td style={{ textAlign: 'center', padding: '10px 12px', fontSize: 12, fontWeight: 700, color: '#16a34a' }}>{m.tp}</td>
                    <td style={{ textAlign: 'center', padding: '10px 12px', fontSize: 12, fontWeight: 700, color: '#dc2626' }}>{m.fp}</td>
                    <td style={{ textAlign: 'center', padding: '10px 12px', fontSize: 12, fontWeight: 700, color: '#d97706' }}>{m.fn}</td>
                    <td style={{ textAlign: 'center', padding: '10px 12px', fontSize: 12, color: '#64748b' }}>{pct(m.precision)}</td>
                    <td style={{ textAlign: 'center', padding: '10px 12px', fontSize: 12, color: '#64748b' }}>{pct(m.recall)}</td>
                    <td style={{ textAlign: 'center', padding: '10px 12px' }}>
                      <span style={{
                        fontSize: 12, fontWeight: 700, padding: '2px 8px', borderRadius: 999,
                        color: scoreColor(m.f1), background: scoreBg(m.f1),
                      }}>
                        {pct(m.f1)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Gold standard list */}
        <div>
          <p style={{ fontSize: 11, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>
            Gold standard deviations
          </p>
          <div style={{ borderRadius: 12, overflow: 'hidden', border: '1px solid #f1f5f9' }}>
            <table style={{ width: '100%', fontSize: 14, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: '#f8fafc', borderBottom: '1px solid #f1f5f9' }}>
                  {[['ID', 'left'], ['Type', 'left'], ['Articles', 'left'], ['Detected', 'center']].map(([h, align]) => (
                    <th key={h} style={{
                      padding: '10px 16px', textAlign: align as 'left' | 'center',
                      fontSize: 11, fontWeight: 600, color: '#64748b',
                      textTransform: 'uppercase', letterSpacing: '0.06em',
                    }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {gold_results.map((g, idx, arr) => (
                  <tr key={g.id} style={{
                    borderBottom: idx < arr.length - 1 ? '1px solid #f8fafc' : 'none',
                    background: g.detected ? 'transparent' : 'rgb(255 251 235 / 0.3)',
                  }}>
                    <td style={{ padding: '10px 16px' }}>
                      <span style={{ fontFamily: 'ui-monospace, monospace', fontSize: 12, fontWeight: 600, color: '#94a3b8' }}>
                        {g.id}
                      </span>
                    </td>
                    <td style={{ padding: '10px 16px' }}><Badge type={g.deviation_type} /></td>
                    <td style={{ padding: '10px 16px', fontSize: 12, color: '#64748b' }}>
                      {g.articles.map(a => `Art. ${a}`).join(', ')}
                    </td>
                    <td style={{ padding: '10px 16px', textAlign: 'center' }}>
                      <DetectedPill detected={g.detected} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  )
}

export function ResultsDisplay({ results }: { results: Results }) {
  return (
    <div>
      {results.results.map(uc => (
        <UseCaseSection key={uc.use_case} uc={uc} />
      ))}
    </div>
  )
}
