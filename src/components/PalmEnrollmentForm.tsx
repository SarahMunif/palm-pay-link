import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { toast } from '@/hooks/use-toast';
import { palmFeatureExtractor } from '@/utils/palmFeatureExtractor';
import { User, CreditCard, DollarSign, Upload, Check } from 'lucide-react';

interface EnrollmentData {
  nationalId: string;
  phoneNumber: string;
  fullName: string;
  cardName: string;
  moneyAmount: string;
  palmImage: File | null;
}

interface PalmEnrollmentFormProps {
  onComplete: (data: EnrollmentData) => void;
}

export const PalmEnrollmentForm = ({ onComplete }: PalmEnrollmentFormProps) => {
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState<EnrollmentData>({
    nationalId: '',
    phoneNumber: '',
    fullName: '',
    cardName: '',
    moneyAmount: '',
    palmImage: null,
  });
  const [dragActive, setDragActive] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const validateStep1 = () => {
    const { nationalId, phoneNumber, fullName, cardName, moneyAmount } = formData;
    
    if (!nationalId || nationalId.length !== 10 || !/^\d{10}$/.test(nationalId)) {
      toast({
        title: "Invalid National ID",
        description: "National ID must be exactly 10 digits",
        variant: "destructive",
      });
      return false;
    }
    
    if (!phoneNumber || phoneNumber.length < 10) {
      toast({
        title: "Invalid Phone Number",
        description: "Please enter a valid phone number",
        variant: "destructive",
      });
      return false;
    }
    
    if (!fullName.trim()) {
      toast({
        title: "Full Name Required",
        description: "Please enter your full name",
        variant: "destructive",
      });
      return false;
    }
    
    if (!cardName.trim()) {
      toast({
        title: "Card Name Required",
        description: "Please enter the card name",
        variant: "destructive",
      });
      return false;
    }
    
    if (!moneyAmount || parseFloat(moneyAmount) <= 0) {
      toast({
        title: "Invalid Amount",
        description: "Please enter a valid money amount",
        variant: "destructive",
      });
      return false;
    }
    
    return true;
  };

  const handleNextStep = async () => {
    if (step === 1 && validateStep1()) {
      setStep(2);
    } else if (step === 2 && formData.palmImage) {
      setIsProcessing(true);
      
      try {
        toast({
          title: "Processing Palm...",
          description: "Extracting biometric features, please wait...",
        });

        // Extract palm features using AI
        const palmFeatures = await palmFeatureExtractor.extractFeatures(formData.palmImage);
        
        // Save enrollment data with palm features
        const enrollmentData = {
          enrollmentId: Date.now().toString(),
          nationalId: formData.nationalId,
          phoneNumber: formData.phoneNumber,
          fullName: formData.fullName,
          cardName: formData.cardName,
          moneyAmount: formData.moneyAmount,
          palmFeatures: palmFeatures,
          palmImageData: URL.createObjectURL(formData.palmImage),
        };
        
        const existingData = JSON.parse(localStorage.getItem('palmEnrollments') || '[]');
        existingData.push(enrollmentData);
        localStorage.setItem('palmEnrollments', JSON.stringify(existingData));
        
        toast({
          title: "Enrollment Successful! ✋",
          description: `Palm biometrics registered for ${formData.fullName}`,
        });
        
        onComplete(formData);
        
      } catch (error) {
        console.error('Enrollment error:', error);
        toast({
          title: "Enrollment Failed",
          description: "Failed to process palm biometrics. Please try again.",
          variant: "destructive",
        });
      } finally {
        setIsProcessing(false);
      }
    } else if (step === 2) {
      toast({
        title: "Palm Image Required",
        description: "Please upload your palm image",
        variant: "destructive",
      });
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
        setFormData({ ...formData, palmImage: file });
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
      setFormData({ ...formData, palmImage: e.target.files[0] });
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-md glass-card">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl font-bold gradient-accent bg-clip-text text-transparent">
            Palm Enrollment
          </CardTitle>
          <CardDescription>
            Step {step} of 2: {step === 1 ? 'Personal Information' : 'Palm Registration'}
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-6">
          {step === 1 ? (
            <>
              <div className="space-y-2">
                <Label htmlFor="nationalId" className="flex items-center gap-2">
                  <User className="w-4 h-4" />
                  National ID
                </Label>
                <Input
                  id="nationalId"
                  placeholder="1234567890"
                  value={formData.nationalId}
                  onChange={(e) => {
                    const value = e.target.value.replace(/\D/g, '').slice(0, 10);
                    setFormData({ ...formData, nationalId: value });
                  }}
                  maxLength={10}
                  className="transition-smooth"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="phoneNumber">Phone Number</Label>
                <Input
                  id="phoneNumber"
                  placeholder="+1 (555) 123-4567"
                  value={formData.phoneNumber}
                  onChange={(e) => setFormData({ ...formData, phoneNumber: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="fullName">Full Name</Label>
                <Input
                  id="fullName"
                  placeholder="John Doe"
                  value={formData.fullName}
                  onChange={(e) => setFormData({ ...formData, fullName: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="cardName" className="flex items-center gap-2">
                  <CreditCard className="w-4 h-4" />
                  Card Name
                </Label>
                <Input
                  id="cardName"
                  placeholder="Premium Gold Card"
                  value={formData.cardName}
                  onChange={(e) => setFormData({ ...formData, cardName: e.target.value })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="moneyAmount" className="flex items-center gap-2">
                  <DollarSign className="w-4 h-4" />
                  Initial Amount
                </Label>
                <Input
                  id="moneyAmount"
                  type="number"
                  placeholder="100.00"
                  value={formData.moneyAmount}
                  onChange={(e) => setFormData({ ...formData, moneyAmount: e.target.value })}
                  min="0"
                  step="0.01"
                />
              </div>
            </>
          ) : (
            <div className="space-y-6">
              <div
                className={`palm-upload-area ${dragActive ? 'active' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <div className="text-center space-y-4">
                  {formData.palmImage ? (
                    <div className="space-y-4">
                      <div className="w-32 h-32 mx-auto rounded-lg overflow-hidden border-2 border-primary">
                        <img
                          src={URL.createObjectURL(formData.palmImage)}
                          alt="Palm preview"
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <div className="flex items-center justify-center gap-2 text-success">
                        <Check className="w-5 h-5" />
                        <span className="font-medium">Palm image uploaded</span>
                      </div>
                    </div>
                  ) : (
                    <>
                      <Upload className="w-16 h-16 mx-auto text-muted-foreground" />
                      <div>
                        <p className="text-lg font-medium">Upload your palm image</p>
                        <p className="text-sm text-muted-foreground">
                          Drag and drop or click to select
                        </p>
                      </div>
                    </>
                  )}
                  
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileUpload}
                    className="hidden"
                    id="palmImageUpload"
                  />
                  <Button
                    variant="glass"
                    asChild
                    className="w-full"
                  >
                    <label htmlFor="palmImageUpload" className="cursor-pointer">
                      {formData.palmImage ? 'Change Image' : 'Select Image'}
                    </label>
                  </Button>
                </div>
              </div>
              
              <div className="p-4 glass-card rounded-lg">
                <p className="text-sm text-muted-foreground text-center">
                  Place your palm flat against a contrasting background and ensure good lighting for best results.
                </p>
              </div>
            </div>
          )}

          <div className="flex gap-3">
            {step === 2 && (
              <Button
                variant="glass"
                onClick={() => setStep(1)}
                className="flex-1"
              >
                Back
              </Button>
            )}
            <Button
              variant="premium"
              onClick={handleNextStep}
              className="flex-1"
              size="lg"
              disabled={isProcessing || (step === 2 && !formData.palmImage)}
            >
              {isProcessing ? 'Processing...' : (step === 1 ? 'Next' : 'Complete Enrollment')}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};