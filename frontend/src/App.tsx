import { useState } from 'react'
import './App.css'
import { Input } from './components/ui/input'
import { Button } from './components/ui/button'
import { Card, CardContent, CardHeader, CardDescription, CardTitle } from './components/ui/card'
import { Globe, Shield, ArrowRight, Loader2,CodeXml } from 'lucide-react'
import { AnalysisResult } from './components/AnalysisResult'
import { Toaster } from './components/ui/toaster'
import { toast } from 'sonner'
import { fetchAnalysis, type BackendResponse, type AnalysisApiResponse } from './lib/api'
import DarkVeil from './components/DarkVeil';

  


export type AnalysisData = {
  address: string
  fraudProbability: number // 0..100
  confidence: number // 0..100
  riskLevel: 'LOW RISK' | 'MEDIUM RISK' | 'HIGH RISK'
  addresses: string[]
  transactions: BackendResponse['fraudulent_transactions']
  warning?: string
  numTransactions: number
}

// Simple, editable defaults (do not hard-code analysis logic here)
const DEFAULTS = {
  placeholderAddress: '0x0000000000000000000000000000000000000000',
}

function App() {
  const [address, setAddress] = useState('')
  const [result, setResult] = useState<AnalysisData | null>(null)
  const [analyzing, setAnalyzing] = useState(false)

  const isValidEthereumAddress = (addr: string) => /^0x[a-fA-F0-9]{40}$/.test(addr)

  const handleAnalyze = async () => {
    // Placeholder: replace with real analysis wiring.
    // Keep values configurable and easy to update.
    if (!isValidEthereumAddress(address)) {
      toast.error('Enter a valid Ethereum address (0x + 40 hex)')
      return
    }
    try {
      setAnalyzing(true)
    const apiRes: AnalysisApiResponse = await fetchAnalysis(address)
    console.log('API response:', apiRes)
    if (!apiRes.ok) {
        const mapped: AnalysisData = {
          address,
          fraudProbability: 0,
          confidence: 0,
          riskLevel: 'LOW RISK',
          addresses: [],
          transactions: [],
      warning: apiRes.warning,
      numTransactions: 0,
        }
        setResult(mapped)
        return
      }
    const data = apiRes.data
    // Probability reflects fraud likelihood: if flagged as fraud, use confidence%; otherwise 0%
    const confidencePct = Math.round((data.confidence ?? 0) * 100)
    const probability = data.is_fraud ? confidencePct : 0
      const risk: AnalysisData['riskLevel'] = probability >= 75 ? 'HIGH RISK' : probability >= 35 ? 'MEDIUM RISK' : 'LOW RISK'
      const mapped: AnalysisData = {
        address,
        fraudProbability: probability,
        confidence: confidencePct,
        riskLevel: risk,
        addresses: Array.isArray(data.addresses_involved) ? data.addresses_involved : [],
        transactions: Array.isArray(data.fraudulent_transactions) ? data.fraudulent_transactions : [],
  numTransactions: typeof data.num_transactions === 'number' ? data.num_transactions : (Array.isArray(data.fraudulent_transactions) ? data.fraudulent_transactions.length : 0),
      }
      setResult(mapped)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Analysis failed'
      toast.error(message)
      setResult(null)
    } finally {
      setAnalyzing(false)
    }
  }

  return (
    <div className="app-root min-h-screen flex flex-col items-center bg-background text-foreground">
      <DarkVeil
      speed={1.0}
      scanlineFrequency={0.0}
      scanlineIntensity={0.0}
      />
      {/* Top navigation bar */}
      <nav className="w-full nav-bar">
        <div className="nav-inner max-w-6xl mx-auto px-6 flex items-center justify-between h-16">
          <div className="flex items-center gap-3 z-30">
            <div className="relative">
              <Globe className="h-6 w-6" />
              <span className="absolute -right-1 -top-1 inline-block size-2 rounded-full bg-primary" />
            </div>
            <div className="leading-tight">
              <span className="block font-semibold tracking-tight">EtherGard</span>
              <span className="block text-[10px] text-muted-foreground -mt-0.5">Ethereum Fraud Detection</span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <Button  variant="outline" size="sm" className="backdrop-blur border-white/15 bg-background/30">
              <CodeXml/>
              <a href="http://3.142.201.165:8000/docs" target="_blank" rel="noreferrer">API</a>
            </Button>
          </div>
        </div>
      </nav>
      
      {/* Hero */}
      <header className="w-full">
        
        <div className="hero max-w-4xl mx-auto px-6 py-20 text-center">
          <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-card/40 px-3 py-1 text-xs text-muted-foreground backdrop-blur">
            <Shield className="h-3.5 w-3.5 text-primary" />
            Real-time risk insights
          </div>
          <h1 className="hero-title mt-4 text-balance text-4xl md:text-6xl font-extrabold leading-tight bg-gradient-to-br from-primary/90 via-foreground to-foreground/70 bg-clip-text text-transparent">
            Scan Ethereum Addresses for Fraudulent Transactions
          </h1>
          <p className="hero-sub mt-4 text-base md:text-lg text-muted-foreground">
            Paste an address and get a quick fraud risk snapshot. No wallet connection required.
          </p>

          {/* Input card */}
          <Card id="analyze" className="hero-card mt-10 mx-auto w-full max-w-3xl border-white/10 bg-card/60 backdrop-blur-xl">
            <CardHeader className="pb-3">
              <CardTitle>Address Scanner</CardTitle>
              <CardDescription>Enter an Ethereum address to begin analysis</CardDescription>
            </CardHeader>
            <CardContent className="pt-0 p-4 sm:p-6 sm:pt-0">
              <div className="flex flex-col sm:flex-row gap-3 items-center">
                    <Input
                    value={address}
                    onChange={(e) => setAddress((e.target as HTMLInputElement).value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
                    placeholder={DEFAULTS.placeholderAddress}
                    className="flex-1 bg-sidebar border-2 focus-visible:ring-0 px-3 py-2 border-sidebar-accent/30 hover:border-sidebar-accent focus:border-sidebar-accent"
                  />
                <Button size="lg" onClick={handleAnalyze} disabled={analyzing} aria-busy={analyzing} className="shrink-0 group min-w-28">
                  {analyzing ? (
                    <span className="inline-flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Analyzing
                    </span>
                  ) : (
                    <span className="inline-flex items-center">Analyze<ArrowRight className="ml-1 h-4 w-4 transition-transform group-hover:translate-x-0.5" /></span>
                  )}
                </Button>
              </div>
              
            </CardContent>
          </Card>
        </div>
      </header>

      <main className="w-full max-w-4xl mx-auto px-6 py-8">
        {analyzing && (
          <Card className="border-white/10 bg-card/60 backdrop-blur-xl">
            <CardContent className="p-6">
              <div className="h-4 w-36 rounded bg-white/10 animate-pulse" />
              <div className="mt-6 h-3 w-full rounded bg-white/10 animate-pulse" />
              <div className="mt-3 h-3 w-5/6 rounded bg-white/10 animate-pulse" />
              <div className="mt-8 h-24 w-full rounded-md bg-white/10 animate-pulse" />
            </CardContent>
          </Card>
        )}
        {result && !analyzing && (
          <div className="result-animate">
            <AnalysisResult data={result} />
          </div>
        )}
      </main>

      <footer className="w-full mt-auto py-8">
        <div className="max-w-4xl mx-auto px-6 text-center text-sm text-muted-foreground">
          © {new Date().getFullYear()} EtherGard
        </div>
      </footer>
      <Toaster richColors />
    </div>
  )
}

export default App
