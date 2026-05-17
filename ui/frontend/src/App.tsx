import type { CSSProperties } from 'react'
import * as Tabs from '@radix-ui/react-tabs'
import { HistoryTab } from './components/HistoryTab'
import { RunTab } from './components/RunTab'
import { ShieldCheck, Play, History } from 'lucide-react'

/* Guaranteed container — does NOT rely on Tailwind named sizes */
const CONTAINER: CSSProperties = {
  maxWidth: 1400,
  margin: '0 auto',
  paddingLeft: 56,
  paddingRight: 56,
  width: '100%',
}

export default function App() {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#f1f5f9' }}>
      <Tabs.Root defaultValue="run" style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>

        {/* ── Shell header ──────────────────────────────────────────────── */}
        <header
          style={{
            background: '#ffffff',
            borderBottom: '1px solid #e2e8f0',
            boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.06)',
            position: 'sticky',
            top: 0,
            zIndex: 20,
          }}
        >
          <div style={CONTAINER}>

            {/* Brand row */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: 64 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div
                  style={{
                    width: 36, height: 36,
                    borderRadius: 10,
                    background: 'linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%)',
                    boxShadow: '0 2px 8px rgb(29 78 216 / 0.35)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                  }}
                >
                  <ShieldCheck size={18} color="#fff" strokeWidth={2} />
                </div>
                <div>
                  <h1 style={{ fontSize: 14, fontWeight: 600, color: '#0f172a', lineHeight: 1.3, margin: 0 }}>
                    GDPR Deviation Pipeline
                  </h1>
                  <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.3, margin: 0 }}>
                    TUM NLP for IS — SS26 Praktikum
                  </p>
                </div>
              </div>

              {/* Use-case badge */}
              <div
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '4px 12px',
                  borderRadius: 999,
                  background: '#eff6ff',
                  border: '1px solid #bfdbfe',
                  color: '#1d4ed8',
                  fontSize: 12,
                  fontWeight: 500,
                }}
              >
                <span
                  className="pulse-dot"
                  style={{ width: 6, height: 6, borderRadius: 999, background: '#22c55e', display: 'inline-block' }}
                />
                Hetzner · Zalando · Trade Republic
              </div>
            </div>

            {/* Tab nav strip */}
            <Tabs.List style={{ display: 'flex', gap: 0, marginBottom: -1 }}>
              {[
                { value: 'run',     label: 'New Run',  Icon: Play    },
                { value: 'history', label: 'History',  Icon: History },
              ].map(({ value, label, Icon }) => (
                <Tabs.Trigger
                  key={value}
                  value={value}
                  style={{ all: 'unset', cursor: 'pointer' }}
                  className="tab-trigger"
                  data-value={value}
                >
                  {/* We use a wrapper span that gets the visual styling */}
                  <span className="tab-inner" data-value={value}>
                    <Icon size={13} strokeWidth={2} className="tab-icon" />
                    {label}
                  </span>
                </Tabs.Trigger>
              ))}
            </Tabs.List>
          </div>
        </header>

        {/* ── Page content ──────────────────────────────────────────────── */}
        <main style={{ flex: 1, ...CONTAINER, paddingTop: 36, paddingBottom: 48, boxSizing: 'border-box' }}>
          <Tabs.Content value="run" style={{ outline: 'none' }}>
            <RunTab />
          </Tabs.Content>
          <Tabs.Content value="history" style={{ outline: 'none' }}>
            <HistoryTab />
          </Tabs.Content>
        </main>

      </Tabs.Root>
    </div>
  )
}
