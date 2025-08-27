import { Copy, CheckCircle, AlertTriangle, XCircle } from 'lucide-react'
import { Button } from './ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Progress } from './ui/progress'
import { Badge } from './ui/badge'
import { toast } from 'sonner'
import type { AnalysisData } from '../App'
import { ExternalLink } from 'lucide-react'

interface AnalysisResultProps {
  data: AnalysisData
}

// Configuration for easy editing
const RESULT_CONFIG = {
  title: 'Analysis Result',
  labels: {
    address: 'Address',
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
  const { address, confidence, riskLevel, addresses, transactions, numTransactions } = data
  // When backend returns a warning (e.g., Etherscan 404 no transactions), prefer showing it over predictions
  const warning: string | undefined = (data as unknown as { warning?: string })?.warning
  
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
  const safeAddresses = Array.isArray(addresses) ? addresses : []
  const safeTransactions = Array.isArray(transactions) ? transactions : []
  const safeNumTransactions = typeof numTransactions === 'number' ? numTransactions : safeTransactions.length
  // Show a yellow warning card instead of predictions if present
  if (warning) {
    const WarningIcon = AlertTriangle
    return (
      <Card className="overflow-hidden border-white/10 bg-card/60  dark:bg-yellow-950/20 backdrop-blur-xl">
        <CardHeader className="text-center">
          <CardTitle>{RESULT_CONFIG.title}</CardTitle>
          <CardDescription>
            <Badge variant="secondary" className="bg-yellow-500/20 text-yellow-800 dark:text-yellow-300 border-yellow-500/30">
              <WarningIcon className="mr-1 h-3 w-3" />
              Warning
            </Badge>
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-medium">{RESULT_CONFIG.labels.address}</label>
            <div className="flex items-center gap-2 p-3 bg-muted rounded-md">
              <code className="flex-1 text-sm font-mono">{`${address.slice(0, 6)}...${address.slice(-4)}`}</code>
              <Button variant="ghost" size="sm" onClick={() => navigator.clipboard.writeText(address)}>
                <Copy className="h-4 w-4" />
              </Button>
            </div>
          </div>
          <div className="rounded-md border border-yellow-500/30 bg-yellow-100/60 dark:bg-yellow-900/30 p-4 text-sm text-yellow-900 dark:text-yellow-200 flex items-start gap-2">
            <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
            <p>{warning}</p>
          </div>
        </CardContent>
      </Card>
    )
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
            <label className="text-sm font-medium">fraud Probability</label>
            <span className="text-sm font-medium">{confidence}%</span>
          </div>
          <div className="relative">
            <Progress value={confidence} className="h-2" />
            <div 
              className={`absolute top-0 left-0 h-2 rounded-full ${riskColor}`}
              style={{ width: `${confidence}%`, transition: 'width 500ms ease' }}
            />
          </div>
        </div>

        {/* Transactions count */}
        <div className="space-y-1 animate-in fade-in slide-in-from-left-2 duration-300 delay-250">
          <div className="flex justify-between items-center">
            <label className="text-sm font-medium">Transactions Analyzed</label>
            <span className="text-sm font-medium">{safeNumTransactions}</span>
          </div>
        </div>

    {/* Addresses involved */}
        <div className="space-y-3 animate-in fade-in duration-300 delay-200">
          <label className="text-sm font-medium">Addresses Mentioned</label>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {safeAddresses.map((addr) => (
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
    {safeTransactions.length ? (
          <div className="space-y-3 animate-in fade-in duration-300 delay-200">
            <label className="text-sm font-medium">Flagged Transactions</label>
            <div className="space-y-2">
      {safeTransactions.map((tx) => (
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