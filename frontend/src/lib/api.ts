export type BackendTransaction = {
  address: string
  confidence: number // 0..1
  transaction_hash: string
  transaction_type: 'incoming' | 'outgoing' | string
  addresses_involved: string[]
}

export type BackendResponse = {
  fraudulent_transactions: BackendTransaction[]
  confidence: number // 0..1
  is_fraud: boolean
  addresses_involved: string[]
  // populated by backend processAdress endpoint; optional for type safety
  prediction_id?: string
}

const API_URL = import.meta.env.VITE_API_URL as string | undefined

// Derive API host (origin) from the process endpoint URL
const DEFAULT_PROCESS_URL = 'http://localhost:8000/api/processAdress'
const PROCESS_URL = API_URL ?? DEFAULT_PROCESS_URL
const API_HOST = new URL(PROCESS_URL).origin

export async function fetchAnalysis(address: string, init?: RequestInit): Promise<BackendResponse> {
  const url = `${PROCESS_URL}?address=${encodeURIComponent(address)}`
  const res = await fetch(url, {
    method: 'Post',
    headers: { 'Accept': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${text || res.statusText}`)
  }
  const data = (await res.json()) as BackendResponse
  return data
}

export type ReportStats = { valid_count: number; invalid_count: number; total_count: number }
export type ReportItem = { user_id: string; is_valid: boolean; note?: string | null; created_at?: string | null }

export async function submitReport(params: { prediction_id: string; user_id: string; is_valid: boolean; note?: string }): Promise<{ report_id: string; stats: ReportStats }>{
  const res = await fetch(`${API_HOST}/api/reports`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Submit report failed (${res.status}): ${text || res.statusText}`)
  }
  return (await res.json()) as { report_id: string; stats: ReportStats }
}

export async function fetchReports(prediction_id: string): Promise<{ prediction_id: string; stats: ReportStats; reports: ReportItem[] }>{
  const res = await fetch(`${API_HOST}/api/reports/${encodeURIComponent(prediction_id)}`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Fetch reports failed (${res.status}): ${text || res.statusText}`)
  }
  return (await res.json()) as { prediction_id: string; stats: ReportStats; reports: ReportItem[] }
}

// New: resolve latest prediction by address so we can always enable reporting even when prediction_id is missing in analysis response
export type DbPrediction = { id: string; address: string; confidence: number; is_fraud: boolean }

export async function fetchPredictionByAddress(address: string): Promise<DbPrediction> {
  const res = await fetch(`${API_HOST}/api/predictions/${encodeURIComponent(address)}`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Fetch prediction by address failed (${res.status}): ${text || res.statusText}`)
  }
  return (await res.json()) as DbPrediction
}
