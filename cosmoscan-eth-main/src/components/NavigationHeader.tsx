import React from 'react';
import { Shield, Search, Info, Code } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function NavigationHeader() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/20 bg-black/40 backdrop-blur-md supports-[backdrop-filter]:bg-black/30">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Shield className="h-8 w-8 text-blue-400" />
              <div className="absolute inset-0 animate-cosmic-pulse">
                <Shield className="h-8 w-8 text-purple-400 opacity-50" />
              </div>
            </div>
            <div>
              <h1 className="text-xl font-space font-bold bg-gradient-to-r from-blue-300 to-purple-300 bg-clip-text text-transparent">
                CosmoGuard
              </h1>
              <p className="text-xs text-gray-300">Ethereum Fraud Detection</p>
            </div>
          </div>

          {/* Navigation */}
          <nav className="hidden md:flex items-center space-x-1">
            <Button variant="ghost" size="sm" className="font-medium text-gray-200 hover:text-white hover:bg-white/10">
              <Search className="h-4 w-4 mr-2" />
              Scanner
            </Button>
            <Button variant="ghost" size="sm" className="font-medium text-gray-200 hover:text-white hover:bg-white/10">
              <Info className="h-4 w-4 mr-2" />
              About
            </Button>
            <Button variant="ghost" size="sm" className="font-medium text-gray-200 hover:text-white hover:bg-white/10">
              <Code className="h-4 w-4 mr-2" />
              API
            </Button>
          </nav>

          {/* Mobile menu button */}
          <Button variant="outline" size="sm" className="md:hidden border-white/30 text-gray-200 hover:bg-white/10">
            <Search className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </header>
  );
}