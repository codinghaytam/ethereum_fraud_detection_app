import React, { useState } from 'react';
import { Search, Radar, Copy, Clock } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';

interface ScanResult {
  address: string;
  fraudScore: number;
  confidence: number;
  status: 'safe' | 'suspicious' | 'fraudulent' | 'unknown';
  timestamp: Date;
}

export default function AddressScanner() {
  const [address, setAddress] = useState('');
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState('');
  const [recentScans] = useState<ScanResult[]>([
    {
      address: '0x742d35Cc6523C0532925a3b8D45C14394667e6ad',
      fraudScore: 15,
      confidence: 92,
      status: 'safe',
      timestamp: new Date(Date.now() - 1000 * 60 * 5)
    },
    {
      address: '0x1234567890123456789012345678901234567890',
      fraudScore: 85,
      confidence: 88,
      status: 'fraudulent',
      timestamp: new Date(Date.now() - 1000 * 60 * 15)
    }
  ]);

  const isValidEthereumAddress = (addr: string) => {
    return /^0x[a-fA-F0-9]{40}$/.test(addr);
  };

  const truncateAddress = (addr: string) => {
    return `${addr.slice(0, 6)}...${addr.slice(-4)}`;
  };

  const handleScan = async () => {
    setError('');
    
    if (!address.trim()) {
      setError('Please enter an Ethereum address');
      return;
    }

    if (!isValidEthereumAddress(address)) {
      setError('Please enter a valid Ethereum address');
      return;
    }

    setIsScanning(true);

    // Simulate API call
    setTimeout(() => {
      const mockResult: ScanResult = {
        address,
        fraudScore: Math.floor(Math.random() * 100),
        confidence: 75 + Math.floor(Math.random() * 25),
        status: Math.random() > 0.7 ? 'fraudulent' : Math.random() > 0.5 ? 'suspicious' : 'safe',
        timestamp: new Date()
      };
      
      setScanResult(mockResult);
      setIsScanning(false);
    }, 3000);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'safe': return 'status-safe';
      case 'suspicious': return 'status-warning';
      case 'fraudulent': return 'status-danger';
      default: return 'status-unknown';
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="space-y-8">
      {/* Main Scanner Card */}
      <Card className="cosmic-card p-8">
        <div className="space-y-6">
          <div className="text-center">
            <h2 className="text-2xl font-space font-bold mb-2">Address Scanner</h2>
            <p className="text-muted-foreground">Enter an Ethereum address to begin analysis</p>
          </div>

          {/* Input Section */}
          <div className="flex flex-col md:flex-row gap-4 items-end">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  type="text"
                  placeholder="0x742d35Cc6523C0532925a3b8D45C14394667e6ad"
                  value={address}
                  onChange={(e) => setAddress(e.target.value)}
                  className="pl-10 font-mono text-sm"
                  disabled={isScanning}
                />
              </div>
            </div>
            <Button
              onClick={handleScan}
              disabled={isScanning}
              size="lg"
              className="cosmic-button px-8"
            >
              {isScanning ? (
                <>
                  <Radar className="h-4 w-4 mr-2 animate-radar-sweep" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Search className="h-4 w-4 mr-2" />
                  Scan Address
                </>
              )}
            </Button>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </div>
      </Card>

      {/* Scanning Animation */}
      {isScanning && (
        <Card className="cosmic-card p-8">
          <div className="text-center space-y-6">
            <div className="relative w-24 h-24 mx-auto">
              <div className="absolute inset-0 border-4 border-primary/20 rounded-full" />
              <div className="absolute inset-0 border-4 border-transparent border-t-primary rounded-full animate-radar-sweep" />
              <Radar className="absolute inset-0 m-auto h-8 w-8 text-primary" />
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-2">Analyzing Blockchain Patterns</h3>
              <p className="text-muted-foreground">Scanning transaction history and network connections...</p>
              <p className="text-sm text-muted-foreground mt-2">Estimated time: ~15 seconds</p>
            </div>
          </div>
        </Card>
      )}

      {/* Scan Result */}
      {scanResult && !isScanning && (
        <Card className="cosmic-card p-8 animate-in slide-in-from-bottom-4 duration-500">
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-space font-bold">Analysis Result</h3>
              <Badge className={getStatusColor(scanResult.status)}>
                {scanResult.status.toUpperCase()}
              </Badge>
            </div>

            <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
              <span className="font-mono text-sm">{truncateAddress(scanResult.address)}</span>
              <Button variant="ghost" size="sm" onClick={() => copyToClipboard(scanResult.address)}>
                <Copy className="h-4 w-4" />
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="text-sm font-medium text-muted-foreground">Fraud Score</label>
                <div className="text-3xl font-bold text-primary">{scanResult.fraudScore}/100</div>
              </div>
              <div>
                <label className="text-sm font-medium text-muted-foreground">AI Confidence</label>
                <div className="text-3xl font-bold text-secondary">{scanResult.confidence}%</div>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Recent Scans */}
      {recentScans.length > 0 && (
        <Card className="cosmic-card p-6">
          <h3 className="text-lg font-semibold mb-4">Recent Scans</h3>
          <div className="space-y-3">
            {recentScans.map((scan, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
                <div className="flex items-center space-x-3">
                  <Badge className={getStatusColor(scan.status)}>
                    {scan.status}
                  </Badge>
                  <span className="font-mono text-sm">{truncateAddress(scan.address)}</span>
                </div>
                <div className="flex items-center text-xs text-muted-foreground">
                  <Clock className="h-3 w-3 mr-1" />
                  {Math.floor((Date.now() - scan.timestamp.getTime()) / 1000 / 60)}m ago
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}