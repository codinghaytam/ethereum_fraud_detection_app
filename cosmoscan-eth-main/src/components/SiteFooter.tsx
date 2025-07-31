import React from 'react';
import { Shield } from 'lucide-react';

export default function SiteFooter() {
  return (
    <footer className="border-t border-border/50 bg-card/30 backdrop-blur">
      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Logo and Description */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Shield className="h-6 w-6 text-primary" />
              <span className="font-space font-bold text-lg">CosmoGuard</span>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Advanced Ethereum fraud detection powered by AI. Protecting the crypto galaxy one address at a time.
            </p>
          </div>

          {/* Links */}
          <div className="space-y-4">
            <h4 className="font-semibold">Resources</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li><a href="#" className="hover:text-primary transition-colors">Privacy Policy</a></li>
              <li><a href="#" className="hover:text-primary transition-colors">Terms of Service</a></li>
              <li><a href="#" className="hover:text-primary transition-colors">Contact</a></li>
              <li><a href="#" className="hover:text-primary transition-colors">API Documentation</a></li>
            </ul>
          </div>

          {/* Disclaimer */}
          <div className="space-y-4">
            <h4 className="font-semibold">Important Notice</h4>
            <p className="text-xs text-muted-foreground leading-relaxed">
              This tool provides analysis based on blockchain data patterns. Results are for informational 
              purposes only and should not be considered as financial or legal advice.
            </p>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-border/30 text-center">
          <p className="text-xs text-muted-foreground">
            © 2024 CosmoGuard. Securing the Ethereum galaxy with advanced fraud detection.
          </p>
        </div>
      </div>
    </footer>
  );
}