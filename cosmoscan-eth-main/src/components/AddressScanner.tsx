import React, { useState } from 'react';
import { Search, Radar, Copy, Clock } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import CircularProgressDemo from '@/components/PercentageDisplay'; // Adjust the import based on your CircularProgress component

import { getAddressPrediction } from '@/lib/utils'; // Adjust the import based on your API utility
import { AddressPredictionResponse} from '@/lib/types'; // Adjust the import based on your types
export default function AddressScanner() {
  const [address, setAddress] = useState('');
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState< AddressPredictionResponse| null>(null);
  const [error, setError] = useState('');
  const [recentScans] = useState<AddressPredictionResponse[]>([
    
  ]);
  console.log(scanResult)
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
    try{
      const reponse = await getAddressPrediction(address);
      setScanResult(
        reponse
      )
      setIsScanning(false);
  }catch (err) {
      setError('Failed to fetch prediction. Please try again later.');
      setIsScanning(false);
    }
    
};




  const getStatusColor = (prediction: string) => {
    switch (prediction.toLowerCase()) {
      case 'normal': return 'bg-green-500 hover:bg-green-600 text-white';
      case 'fraud': return 'bg-red-500 hover:bg-red-600 text-white';
      default: return 'bg-gray-500 hover:bg-gray-600 text-white';
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
              <Badge className={getStatusColor(scanResult.prediction)}>
                {scanResult.prediction.toUpperCase()}
                
              </Badge>
            </div>

            <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
              <span className="font-mono text-sm">{truncateAddress(scanResult.address)}</span>
              <Button variant="ghost" size="sm" onClick={() => copyToClipboard(scanResult.address)}>
                <Copy className="h-4 w-4" />
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="text-center">
                <h4 className="text-sm font-medium mb-2 text-muted-foreground">Fraud Probability</h4>
                <CircularProgressDemo value={scanResult.fraud_probability*100} />
              </div>
              <div className="text-center">
                <h4 className="text-sm font-medium mb-2 text-muted-foreground">Confidence</h4>
                <CircularProgressDemo value={scanResult.confidence*100}  />
              </div>
            </div>

            {/* Fraud Explanations - only show if address is fraudulent */}
            {scanResult.is_fraud && scanResult.explanations && scanResult.explanations.length > 0 && (
              <div className="space-y-4">
                <h4 className="text-lg font-semibold text-red-600">⚠️ Fraud Indicators</h4>
                <div className="space-y-4">
                  {scanResult.explanations.map((explanation, index) => (
                    <div key={index} className="p-4 border border-red-200 bg-red-50 rounded-lg">
                      <h5 className="font-semibold text-red-800 mb-2">{explanation.title}</h5>
                      <p className="text-red-700 text-sm mb-3">{explanation.description}</p>
                      {explanation.examples && explanation.examples.length > 0 && (
                        <div className="space-y-1">
                          <p className="font-medium text-red-800 text-sm">Common examples:</p>
                          <ul className="list-disc list-inside space-y-1 text-red-600 text-sm">
                            {explanation.examples.map((example, exampleIndex) => (
                              <li key={exampleIndex}>{example}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
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
                  <Badge className={getStatusColor(scan.prediction)}>
                    {scan.prediction.toUpperCase()}
                  </Badge>
                  <span className="font-mono text-sm">{truncateAddress(scan.address)}</span>
                </div>
                {/* <div className="flex items-center text-xs text-muted-foreground">
                  <Clock className="h-3 w-3 mr-1" />
                  {Math.floor((Date.now() - scan..getTime()) / 1000 / 60)}m ago
                </div> */}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}