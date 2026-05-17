import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatTs(iso: string | null): string {
  if (!iso) return 'Unknown date'
  const d = new Date(iso)
  return d.toLocaleString('en-DE', {
    year: 'numeric', month: 'short', day: '2-digit',
    hour: '2-digit', minute: '2-digit',
  })
}

export function pct(n: number): string {
  return (n * 100).toFixed(1) + '%'
}
