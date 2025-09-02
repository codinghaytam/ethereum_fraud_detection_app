import { Copy, CheckCircle, AlertTriangle, XCircle } from 'lucide-react'
import { Button } from './ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Progress } from './ui/progress'
import { Badge } from './ui/badge'
import { toast } from 'sonner'
import type { AnalysisData } from '../App'
import { ExternalLink } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { fetchReports, submitReport, type ReportStats, fetchPredictionByAddress } from '../lib/api'
import { getOrCreateUserId } from '../lib/utils'

interface AnalysisResultProps {
  data: AnalysisData
}

// Configuration for easy editing
const RESULT_CONFIG = {
  title: 'Analysis Result',
  labels: {
    address: 'Address',
    fraudProbability: 'Fraud Probability',
    confidence: 'Confidence'
  },
  riskColors: {
    'LOW RISK': 'bg-green-500',
    'MEDIUM RISK': 'bg-yellow-500', 
    'HIGH RISK': 'bg-red-500'
  },
  riskIcons: {
    'LOW RISK': CheckCircle,
    'MEDIUM RISK': AlertTriangle,
    'HIGH RISK': XCircle
  }
}

export function AnalysisResult({ data }: AnalysisResultProps) {
  const { address, fraudProbability, riskLevel, addresses, transactions, predictionId } = data
  const [note, setNote] = useState('')
  const [stats, setStats] = useState<ReportStats | null>(null)
  const [loadingStats, setLoadingStats] = useState(false)
  const [submitting, setSubmitting] = useState<'valid' | 'invalid' | null>(null)
  const userId = useMemo(() => getOrCreateUserId(), [])

  // New: resolve prediction id by address if missing, so reporting is always available
  const [resolvedPredictionId, setResolvedPredictionId] = useState<string | undefined>(predictionId)
  const [attemptedAnalysisFallback, setAttemptedAnalysisFallback] = useState(false)

  useEffect(() => {
    let active = true
    async function resolveId() {
      try {
        if (predictionId) {
          if (!active) return
          setResolvedPredictionId(predictionId)
          return
        }
        if (!address) return
        // Try to resolve from DB by address first
        const pred = await fetchPredictionByAddress(address)
        if (!active) return
        if (pred?.id) {
          setResolvedPredictionId(pred.id)
          return
        }
      } catch {
        // ignore and try fallback below
      }
      // Fallback: trigger a fresh analysis once to ensure we get a prediction_id
      try {
        if (!attemptedAnalysisFallback && address) {
          setAttemptedAnalysisFallback(true)
          const { fetchAnalysis } = await import('../lib/api')
          const fresh = await fetchAnalysis(address)
          if (!active) return
          if (fresh?.prediction_id) {
            setResolvedPredictionId(fresh.prediction_id)
          }
        }
      } catch {
        // swallow; UI remains without reporting if backend unavailable
      }
    }
    resolveId()
    return () => { active = false }
  }, [predictionId, address, attemptedAnalysisFallback])

  useEffect(() => {
    let active = true
    async function load() {
      if (!resolvedPredictionId) return
      try {
        setLoadingStats(true)
        const res = await fetchReports(resolvedPredictionId)
        if (!active) return
        setStats(res.stats)
      } catch {
        // Silent; UI works without stats
      } finally {
        if (active) setLoadingStats(false)
      }
    }
    load()
    return () => {
      active = false
    }
  }, [resolvedPredictionId])

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      toast.success('Address copied to clipboard')
    } catch {
      toast.error('Failed to copy address')
    }
  }

  const formatAddress = (addr: string) => {
    return `${addr.slice(0, 6)}...${addr.slice(-4)}`
  }

  const RiskIcon = RESULT_CONFIG.riskIcons[riskLevel]
  const riskColor = RESULT_CONFIG.riskColors[riskLevel]

  const canReport = Boolean(resolvedPredictionId)

  async function handleReport(is_valid: boolean) {
    if (!resolvedPredictionId) {
      toast.error('Prediction ID missing; cannot submit report')
      return
    }
    try {
      setSubmitting(is_valid ? 'valid' : 'invalid')
      const res = await submitReport({ prediction_id: resolvedPredictionId, user_id: userId, is_valid, note: note || undefined })
      setStats(res.stats)
      toast.success('Thanks for your feedback')
      setNote('')
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to submit report'
      toast.error(msg)
    } finally {
      setSubmitting(null)
    }
  }

  return (
    <Card className="overflow-hidden border-white/10 bg-card/60 backdrop-blur-xl">
      <CardHeader className="text-center animate-in fade-in slide-in-from-top-2 duration-300">
        <CardTitle>{RESULT_CONFIG.title}</CardTitle>
        <CardDescription>
          <Badge variant={riskLevel === 'LOW RISK' ? 'default' : riskLevel === 'MEDIUM RISK' ? 'secondary' : 'destructive'}>
            <RiskIcon className="mr-1 h-3 w-3" />
            {riskLevel}
          </Badge>
        </CardDescription>
      </CardHeader>
  <CardContent className="space-y-8">
        {/* Address Display */}
        <div className="space-y-2 animate-in fade-in slide-in-from-left-2 duration-300 delay-100">
          <label className="text-sm font-medium">{RESULT_CONFIG.labels.address}</label>
          <div className="flex items-center gap-2 p-3 bg-muted rounded-md">
            <code className="flex-1 text-sm font-mono">
              {formatAddress(address)}
            </code>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => copyToClipboard(address)}
            >
              <Copy className="h-4 w-4" />
            </Button>
          </div>
        </div>


        {/* Confidence */}
        <div className="space-y-2 animate-in fade-in slide-in-from-bottom-2 duration-300 delay-200">
          <div className="flex justify-between items-center">
            <label className="text-sm font-medium">Fraud Probability</label>
            <span className="text-sm font-medium">{fraudProbability}%</span>
          </div>
          <div className="relative">
            <Progress value={fraudProbability} className="h-2" />
            <div
              className={`absolute top-0 left-0 h-2 rounded-full ${riskColor}`}
              style={{ width: `${fraudProbability}%`, transition: 'width 500ms ease' }}
            />
          </div>
        </div>

        {/* Addresses involved */}
        <div className="space-y-3 animate-in fade-in duration-300 delay-200">
          <label className="text-sm font-medium">Addresses Mentioned</label>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {addresses.map((addr) => (
              <a
                key={addr}
                href={`https://etherscan.io/address/${addr}`}
                target="_blank"
                rel="noreferrer"
                className="group rounded-lg border border-white/10 bg-background/50 p-3 hover:bg-background/70 transition flex items-center justify-between"
              >
                <code className="text-xs sm:text-sm truncate mr-2">{formatAddress(addr)}</code>
                <ExternalLink className="h-4 w-4 opacity-60 group-hover:opacity-100" />
              </a>
            ))}
          </div>
        </div>

        {/* Transactions */}
        {transactions?.length ? (
          <div className="space-y-3 animate-in fade-in duration-300 delay-200">
            <label className="text-sm font-medium">Flagged Transactions</label>
            <div className="space-y-2">
              {transactions.map((tx) => (
                <div key={tx.transaction_hash} className="rounded-lg border border-white/10 bg-background/50 p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <Badge variant={tx.transaction_type === 'incoming' ? 'secondary' : 'outline'} className="capitalize">
                        {tx.transaction_type}
                      </Badge>
                      <span className="text-xs text-muted-foreground">{Math.round(tx.confidence * 100)}% conf.</span>
                    </div>
                    <a
                      href={`https://etherscan.io/tx/${tx.transaction_hash}`}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                    >
                      View on Etherscan <ExternalLink className="h-3.5 w-3.5" />
                    </a>
                  </div>
                  <div className="mt-2 text-xs text-muted-foreground break-all">
                    {tx.addresses_involved.join(', ')}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {/* Reporting section */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Community validation</label>
            {stats ? (
              <div className="text-xs text-muted-foreground">
                <span className="mr-3">Valid: <span className="font-medium text-green-500">{stats.valid_count}</span></span>
                <span>Invalid: <span className="font-medium text-red-500">{stats.invalid_count}</span></span>
              </div>
            ) : (
              <div className="text-xs text-muted-foreground">{loadingStats ? 'Loading…' : 'No feedback yet'}</div>
            )}
          </div>
          <textarea
            value={note}
            onChange={(e) => setNote(e.currentTarget.value)}
            placeholder={canReport ? 'Optional: share why you think this is valid or not' : 'Run an analysis to enable reporting'}
            disabled={!canReport}
            className="w-full min-h-20 rounded-md border border-white/10 bg-background/50 p-2 text-sm outline-none disabled:opacity-60"
          />
          <div className="flex gap-2">
            <Button
              variant="secondary"
              disabled={!canReport || submitting !== null}
              onClick={() => handleReport(true)}
            >
              {submitting === 'valid' ? 'Submitting…' : 'Mark Valid'}
            </Button>
            <Button
              variant="destructive"
              disabled={!canReport || submitting !== null}
              onClick={() => handleReport(false)}
            >
              {submitting === 'invalid' ? 'Submitting…' : 'Mark Invalid'}
            </Button>
          </div>
        </div>
        {/* Risk Level Indicator */}
        <div className="flex items-center justify-center p-4 border rounded-lg animate-in fade-in zoom-in-95 duration-300 delay-300">
          <div className="text-center">
            <RiskIcon className={`h-8 w-8 mx-auto mb-2 ${
              riskLevel === 'LOW RISK' ? 'text-green-500' : 
              riskLevel === 'MEDIUM RISK' ? 'text-yellow-500' : 
              'text-red-500'
            }`} />
            <p className="font-medium">{riskLevel}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}