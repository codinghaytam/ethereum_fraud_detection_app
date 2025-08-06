// TypeScript interfaces for backend response
export interface FraudExplanation {
  title: string;
  description: string;
  examples: string[];
}

export interface AddressPredictionResponse {
  prediction: string;
  is_fraud: boolean;
  address: string;
  fraud_probability: number;
  total_transactions: number;
  analysis_timestamp: string;
  confidence: number;
  explanations: FraudExplanation[];
}
