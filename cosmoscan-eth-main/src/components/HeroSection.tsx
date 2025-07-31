import React from 'react';
import { Radar, Shield, Zap } from 'lucide-react';

export default function HeroSection() {
  return (
    <section className="relative py-20 overflow-hidden">
     

      <div className="container mx-auto px-6 text-center relative z-10">
        <div className="max-w-4xl mx-auto">
          

          {/* Main headline */}
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-space font-bold mb-6">
            <span className="bg-gradient-to-r from-white via-blue-300 to-purple-300 bg-clip-text text-transparent drop-shadow-lg">
              Scan the Ethereum
            </span>
            <br />
            <span className="bg-gradient-to-r from-purple-300 via-blue-300 to-white bg-clip-text text-transparent drop-shadow-lg">
              Galaxy
            </span>
          </h1>

          {/* Subtitle */}
          <p className="text-xl md:text-2xl text-gray-100 mb-8 max-w-3xl mx-auto leading-relaxed font-medium">
            Detect fraudulent addresses with{' '}
            <span className="text-blue-300 font-bold bg-blue-400/20 px-2 py-1 rounded-lg">AI-powered analysis</span>
          </p>

          {/* Description */}
          <p className="text-lg text-gray-200 max-w-2xl mx-auto mb-12 leading-relaxed font-medium">
            Enter any Ethereum address to analyze its transaction patterns and get fraud risk assessment 
            powered by advanced machine learning algorithms.
          </p>
        </div>
      </div>
      
    </section>
  );
}