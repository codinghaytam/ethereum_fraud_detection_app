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
  num_transactions: number
}

const API_URL = import.meta.env.VITE_API_URL as string | undefined

export type AnalysisOk = { ok: true; data: BackendResponse }
export type AnalysisWarn = { ok: false; warning: string }
export type AnalysisApiResponse = AnalysisOk | AnalysisWarn

export async function fetchAnalysis(address: string, init?: RequestInit): Promise<AnalysisApiResponse> {
  const base = API_URL ?? 'http://localhost:8000/api/processAdress'
  const url = `${base}?address=${encodeURIComponent(address)}`
    try {
      const res = await fetch(url, {
        method: 'Post',
        headers: { 'Accept': 'application/json' },
        ...init,
      })

      // Read the body once and branch on content
      const text = await res.text().catch(() => '')
      let json: unknown = null
      try { json = text ? JSON.parse(text) : null } catch { json = null }

      // Extract warning or detail when present
      const extractWarning = (j: unknown): string | undefined => {
        if (j && typeof j === 'object') {
          const obj = j as { [k: string]: unknown }
          if (typeof obj.warning === 'string') return obj.warning
          if (typeof obj.detail === 'string') return obj.detail
        }
        return undefined
      }

      const warningFromJson = extractWarning(json)

      // Non-OK -> return warning with best-available message
      if (!res.ok) {
        if (warningFromJson) {
          return { ok: false, warning: warningFromJson }
        }
        const fallback = text || res.statusText || 'Request failed'
        return { ok: false, warning: fallback }
      }

      // OK with a warning payload
      if (warningFromJson) {
        return { ok: false, warning: warningFromJson }
      }

      // Otherwise, treat as a normal backend response
      const data = (json ?? { fraudulent_transactions: [] as BackendTransaction[], confidence: 0, is_fraud: false, addresses_involved: [] as string[], num_transactions: 0 }) as BackendResponse
      return { ok: true, data }
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Network error'
      return { ok: false, warning: msg }
    }
}
