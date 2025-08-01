import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { toast } from '@/hooks/use-toast';
import { palmFeatureExtractor } from '@/utils/palmFeatureExtractor';
import { Badge } from '@/components/ui/badge';
import { Upload, CreditCard, CheckCircle, XCircle, Hand, DollarSign, User } from 'lucide-react';

interface PaymentData {
  amount: number;
  isSuccess: boolean;
  userInfo?: any;
  similarity?: number;
}

interface PalmPaymentProps {
  onBack: () => void;
}

export const PalmPayment = ({ onBack }: PalmPaymentProps) => {
  const [paymentImage, setPaymentImage] = useState<File | null>(null);
  const [paymentResult, setPaymentResult] = useState<PaymentData | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const processPayment = async () => {
    if (!paymentImage) {
      toast({
        title: "Palm Image Required",
        description: "Please upload your palm image for verification",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);

    try {
      toast({
        title: "Verifying Palm...",
        description: "Processing biometric authentication...",
      });

      // Use real AI palm verification
      const verificationResult = await palmFeatureExtractor.verifyPalm(paymentImage);
      
      if (verificationResult.isMatch && verificationResult.userInfo) {
        setPaymentResult({
          amount: verificationResult.finalAmount,
          isSuccess: true,
          userInfo: verificationResult.userInfo,
          similarity: verificationResult.similarity,
        });

        toast({
          title: "Payment Authorized! ✋",
          description: `Payment of $${verificationResult.finalAmount.toFixed(2)} processed successfully for ${verificationResult.userInfo.fullName}`,
        });
      } else {
        setPaymentResult({
          amount: 30,
          isSuccess: false,
          similarity: verificationResult.similarity,
        });

        toast({
          title: "Authentication Failed",
          description: `Palm not recognized (${(verificationResult.similarity * 100).toFixed(1)}% match). Please try again or enroll your palm first.`,
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Payment processing error:', error);
      
      setPaymentResult({
        amount: 30,
        isSuccess: false,
      });

      toast({
        title: "Processing Error",
        description: "Failed to process payment. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        setPaymentImage(file);
        setPaymentResult(null); // Reset previous results
      } else {
        toast({
          title: "Invalid File Type",
          description: "Please upload an image file",
          variant: "destructive",
        });
      }
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setPaymentImage(e.target.files[0]);
      setPaymentResult(null); // Reset previous results
    }
  };

  const resetPayment = () => {
    setPaymentImage(null);
    setPaymentResult(null);
    setIsProcessing(false);
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-md glass-card">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl font-bold gradient-accent bg-clip-text text-transparent flex items-center justify-center gap-2">
            <CreditCard className="w-6 h-6 text-primary" />
            Palm Payment
          </CardTitle>
          <CardDescription>
            Secure payment processing with palm biometrics
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-6">
          {/* Payment Amount */}
          <div className="glass-card p-4 rounded-lg text-center">
            <div className="flex items-center justify-center gap-2 mb-2">
              <DollarSign className="w-5 h-5 text-primary" />
              <span className="text-lg font-semibold">Payment Amount</span>
            </div>
            <div className="text-3xl font-bold text-primary">$30.00</div>
            <p className="text-sm text-muted-foreground mt-1">
              Amount may be discounted based on your palm balance
            </p>
          </div>

          {/* Palm Upload Area */}
          {!paymentResult && (
            <div
              className={`palm-upload-area ${dragActive ? 'active' : ''} ${isProcessing ? 'palm-scanner' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <div className="text-center space-y-4">
                {paymentImage ? (
                  <div className="space-y-4">
                    <div className="w-32 h-32 mx-auto rounded-lg overflow-hidden border-2 border-primary">
                      <img
                        src={URL.createObjectURL(paymentImage)}
                        alt="Payment palm preview"
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="flex items-center justify-center gap-2 text-success">
                      <Hand className="w-5 h-5" />
                      <span className="font-medium">Palm image ready for verification</span>
                    </div>
                  </div>
                ) : (
                  <>
                    <Hand className="w-16 h-16 mx-auto text-muted-foreground" />
                    <div>
                      <p className="text-lg font-medium">Upload palm for payment</p>
                      <p className="text-sm text-muted-foreground">
                        Place your registered palm for verification
                      </p>
                    </div>
                  </>
                )}
                
                {isProcessing && (
                  <div className="text-center">
                    <div className="text-primary font-medium">Verifying palm...</div>
                    <div className="text-sm text-muted-foreground">Please wait</div>
                  </div>
                )}
                
                {!isProcessing && (
                  <>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileUpload}
                      className="hidden"
                      id="paymentPalmUpload"
                    />
                    <Button
                      variant="glass"
                      asChild
                      className="w-full"
                    >
                      <label htmlFor="paymentPalmUpload" className="cursor-pointer">
                        {paymentImage ? 'Change Image' : 'Upload Palm Image'}
                      </label>
                    </Button>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Payment Result */}
          {paymentResult && (
            <div className="space-y-4">
              <div className={`glass-card p-4 rounded-lg border-2 ${
                paymentResult.isSuccess ? 'border-success' : 'border-destructive'
              }`}>
                <div className="flex items-center justify-center gap-2 mb-3">
                  {paymentResult.isSuccess ? (
                    <CheckCircle className="w-8 h-8 text-success" />
                  ) : (
                    <XCircle className="w-8 h-8 text-destructive" />
                  )}
                  <span className="text-lg font-semibold">
                    {paymentResult.isSuccess ? 'Payment Successful' : 'Payment Failed'}
                  </span>
                </div>
                
                {paymentResult.isSuccess && paymentResult.userInfo && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-center gap-2">
                      <User className="w-4 h-4" />
                      <span className="font-medium">{paymentResult.userInfo.fullName}</span>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-2xl font-bold text-success">
                        ${paymentResult.amount.toFixed(2)}
                      </div>
                      {paymentResult.amount < 30 && (
                        <div className="text-sm text-muted-foreground">
                          Discount applied: ${(30 - paymentResult.amount).toFixed(2)}
                        </div>
                      )}
                    </div>
                    
                    {paymentResult.similarity && (
                      <div className="flex justify-center">
                        <Badge variant="secondary">
                          {(paymentResult.similarity * 100).toFixed(1)}% Match
                        </Badge>
                      </div>
                    )}
                  </div>
                )}
                
                {!paymentResult.isSuccess && (
                  <div className="text-center text-destructive-foreground">
                    Palm verification failed. Please try again or contact support.
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3">
            <Button
              variant="glass"
              onClick={onBack}
              className="flex-1"
              disabled={isProcessing}
            >
              Back to Enrollment
            </Button>
            
            {!paymentResult ? (
              <Button
                variant="premium"
                onClick={processPayment}
                className="flex-1"
                size="lg"
                disabled={!paymentImage || isProcessing}
              >
                {isProcessing ? 'Processing...' : 'Pay $30.00'}
              </Button>
            ) : (
              <Button
                variant="glass"
                onClick={resetPayment}
                className="flex-1"
              >
                New Payment
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};