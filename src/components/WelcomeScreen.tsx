import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Hand, Shield, Zap, CreditCard, Users, CheckCircle } from 'lucide-react';

interface WelcomeScreenProps {
  onStartEnrollment: () => void;
  onStartPayment: () => void;
}

export const WelcomeScreen = ({ onStartEnrollment, onStartPayment }: WelcomeScreenProps) => {
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-4xl space-y-8">
        {/* Hero Section */}
        <Card className="glass-card text-center">
          <CardHeader className="pb-8">
            <div className="flex justify-center mb-4">
              <div className="p-4 rounded-full gradient-primary glow-effect">
                <Hand className="w-12 h-12 text-primary-foreground" />
              </div>
            </div>
            <CardTitle className="text-4xl font-bold gradient-accent bg-clip-text text-transparent mb-2">
              Palm Payment System
            </CardTitle>
            <CardDescription className="text-lg max-w-2xl mx-auto">
              Secure, contactless payments using advanced palm biometric technology. 
              Your palm is your wallet - no cards, no phones, just you.
            </CardDescription>
          </CardHeader>
          
          <CardContent className="space-y-8">
            {/* Feature Grid */}
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center space-y-3">
                <div className="w-12 h-12 mx-auto rounded-lg bg-primary/10 flex items-center justify-center">
                  <Shield className="w-6 h-6 text-primary" />
                </div>
                <h3 className="font-semibold">Ultra Secure</h3>
                <p className="text-sm text-muted-foreground">
                  Advanced biometric encryption protects your identity
                </p>
              </div>
              
              <div className="text-center space-y-3">
                <div className="w-12 h-12 mx-auto rounded-lg bg-primary/10 flex items-center justify-center">
                  <Zap className="w-6 h-6 text-primary" />
                </div>
                <h3 className="font-semibold">Lightning Fast</h3>
                <p className="text-sm text-muted-foreground">
                  Complete transactions in under 2 seconds
                </p>
              </div>
              
              <div className="text-center space-y-3">
                <div className="w-12 h-12 mx-auto rounded-lg bg-primary/10 flex items-center justify-center">
                  <CreditCard className="w-6 h-6 text-primary" />
                </div>
                <h3 className="font-semibold">No Cards Needed</h3>
                <p className="text-sm text-muted-foreground">
                  Your palm is all you need for secure payments
                </p>
              </div>
            </div>
            
            {/* How It Works */}
            <div className="space-y-6">
              <h3 className="text-xl font-semibold text-center">How It Works</h3>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="glass-card p-4 rounded-lg">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-6 h-6 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-bold">
                      1
                    </div>
                    <span className="font-medium">Enroll Your Palm</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Register your palm biometrics with personal information
                  </p>
                </div>
                
                <div className="glass-card p-4 rounded-lg">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-6 h-6 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-bold">
                      2
                    </div>
                    <span className="font-medium">Upload Palm Image</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Present your palm for verification and payment
                  </p>
                </div>
                
                <div className="glass-card p-4 rounded-lg">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-6 h-6 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-bold">
                      3
                    </div>
                    <span className="font-medium">Instant Payment</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Automatic verification and payment processing
                  </p>
                </div>
              </div>
            </div>
            
            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center max-w-md mx-auto">
              <Button
                variant="premium"
                size="xl"
                onClick={onStartEnrollment}
                className="flex-1"
              >
                <Users className="w-5 h-5 mr-2" />
                Start Enrollment
              </Button>
              
              <Button
                variant="glass"
                size="xl"
                onClick={onStartPayment}
                className="flex-1"
              >
                <CreditCard className="w-5 h-5 mr-2" />
                Make Payment
              </Button>
            </div>
            
            {/* Demo Notice */}
            <div className="glass-card p-4 rounded-lg max-w-md mx-auto">
              <div className="flex items-center gap-2 justify-center text-sm text-muted-foreground">
                <CheckCircle className="w-4 h-4 text-success" />
                <span>Demo System - All data stored locally</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};