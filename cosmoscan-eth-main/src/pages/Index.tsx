import React from 'react';
import SpaceBackground from '@/components/SpaceBackground';
import NavigationHeader from '@/components/NavigationHeader';
import HeroSection from '@/components/HeroSection';
import AddressScanner from '@/components/AddressScanner';
import SiteFooter from '@/components/SiteFooter';
import DarkVeil from '@/components/Darkveil';

const Index = () => {
  return (
    <div className=" bg-background">
      <NavigationHeader />
      <main className="relative z-10">
         <div className="absolute inset-0">
        <DarkVeil />
        </div> 
        {/* Hero section */}
        <section id="about">
          <HeroSection />
        </section>
        
        {/* Scanner section */}
        <section id="scanner" className="py-12">
          <div className="container mx-auto px-6">
            <AddressScanner />
          </div>
        </section>
      </main>
      
      {/* Footer */}
      <SiteFooter />
    </div>
  );
};

export default Index;
