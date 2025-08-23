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
}

const API_URL = import.meta.env.VITE_API_URL as string | undefined



export async function fetchAnalysis(address: string, init?: RequestInit): Promise<BackendResponse> {
  const base = API_URL ?? 'http://3.142.201.165:8000/api/processAdress'
  const url = `${base}?address=${encodeURIComponent(address)}`
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
