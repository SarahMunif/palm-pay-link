import React, { useState } from 'react';
import { WelcomeScreen } from '@/components/WelcomeScreen';
import { PalmEnrollmentForm } from '@/components/PalmEnrollmentForm';
import { PalmPayment } from '@/components/PalmPayment';

type AppState = 'welcome' | 'enrollment' | 'payment';

const Index = () => {
  const [currentState, setCurrentState] = useState<AppState>('welcome');

  const handleEnrollmentComplete = () => {
    setCurrentState('payment');
  };

  const handleBackToWelcome = () => {
    setCurrentState('welcome');
  };

  const renderCurrentScreen = () => {
    switch (currentState) {
      case 'welcome':
        return (
          <WelcomeScreen
            onStartEnrollment={() => setCurrentState('enrollment')}
            onStartPayment={() => setCurrentState('payment')}
          />
        );
      case 'enrollment':
        return (
          <PalmEnrollmentForm
            onComplete={handleEnrollmentComplete}
          />
        );
      case 'payment':
        return (
          <PalmPayment
            onBack={handleBackToWelcome}
          />
        );
      default:
        return <WelcomeScreen onStartEnrollment={() => setCurrentState('enrollment')} onStartPayment={() => setCurrentState('payment')} />;
    }
  };

  return (
    <div className="min-h-screen">
      {renderCurrentScreen()}
    </div>
  );
};

export default Index;
