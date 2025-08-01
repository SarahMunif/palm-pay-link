import { pipeline, env } from '@huggingface/transformers';

// Configure transformers.js
env.allowLocalModels = false;
env.useBrowserCache = true;

interface PalmFeatures {
  features: number[];
  imageHash: string;
}

interface EnrollmentData {
  enrollmentId: string;
  nationalId: string;
  phoneNumber: string;
  fullName: string;
  cardName: string;
  moneyAmount: string;
  palmFeatures: PalmFeatures;
}

class PalmFeatureExtractor {
  private model: any = null;
  private readonly SIMILARITY_THRESHOLD = 0.85;

  async initialize() {
    if (!this.model) {
      console.log('Initializing MobileNetV2 feature extractor...');
      this.model = await pipeline(
        'image-classification',
        'Xenova/mobilenet_v2_1.0_224',
        { device: 'webgpu' }
      );
      console.log('Feature extractor initialized');
    }
    return this.model;
  }

  async preprocessPalmImage(imageFile: File): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        // Create canvas for preprocessing
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        if (!ctx) {
          reject(new Error('Could not get canvas context'));
          return;
        }

        // Resize to 224x224 for MobileNetV2
        canvas.width = 224;
        canvas.height = 224;
        
        // Draw and enhance image
        ctx.drawImage(img, 0, 0, 224, 224);
        
        // Apply basic contrast enhancement (simulating CLAHE)
        const imageData = ctx.getImageData(0, 0, 224, 224);
        const data = imageData.data;
        
        // Simple contrast enhancement
        for (let i = 0; i < data.length; i += 4) {
          // Enhance contrast for each channel
          data[i] = Math.min(255, Math.max(0, (data[i] - 128) * 1.2 + 128));     // R
          data[i + 1] = Math.min(255, Math.max(0, (data[i + 1] - 128) * 1.2 + 128)); // G
          data[i + 2] = Math.min(255, Math.max(0, (data[i + 2] - 128) * 1.2 + 128)); // B
        }
        
        ctx.putImageData(imageData, 0, 0);
        
        // Convert back to image
        const enhancedImg = new Image();
        enhancedImg.onload = () => resolve(enhancedImg);
        enhancedImg.src = canvas.toDataURL();
      };
      
      img.onerror = reject;
      img.src = URL.createObjectURL(imageFile);
    });
  }

  calculateImageHash(imageFile: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const arrayBuffer = e.target?.result as ArrayBuffer;
          const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer);
          const hashArray = Array.from(new Uint8Array(hashBuffer));
          const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
          resolve(hashHex);
        } catch (error) {
          reject(error);
        }
      };
      reader.onerror = reject;
      reader.readAsArrayBuffer(imageFile);
    });
  }

  async extractFeatures(imageFile: File): Promise<PalmFeatures> {
    try {
      console.log('Extracting palm features...');
      
      // Initialize model if needed
      await this.initialize();
      
      // Preprocess image
      const processedImage = await this.preprocessPalmImage(imageFile);
      
      // Extract features using MobileNetV2
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d')!;
      canvas.width = 224;
      canvas.height = 224;
      ctx.drawImage(processedImage, 0, 0);
      
      // Convert to the format expected by the model
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      
      // Get features from the model
      const result = await this.model(imageData);
      
      // Extract feature vector (use the logits/embeddings)
      let features: number[];
      if (Array.isArray(result) && result.length > 0) {
        // Convert classification scores to feature vector
        features = result.map(r => r.score);
      } else {
        // Fallback: create features from image data
        features = this.createFallbackFeatures(canvas);
      }
      
      // Normalize features (L2 normalization)
      const norm = Math.sqrt(features.reduce((sum, val) => sum + val * val, 0));
      const normalizedFeatures = features.map(val => val / norm);
      
      // Calculate image hash
      const imageHash = await this.calculateImageHash(imageFile);
      
      console.log(`Extracted ${normalizedFeatures.length} features`);
      
      return {
        features: normalizedFeatures,
        imageHash
      };
      
    } catch (error) {
      console.error('Error extracting features:', error);
      
      // Fallback feature extraction
      console.log('Using fallback feature extraction...');
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d')!;
      canvas.width = 224;
      canvas.height = 224;
      
      const img = await this.preprocessPalmImage(imageFile);
      ctx.drawImage(img, 0, 0);
      
      const features = this.createFallbackFeatures(canvas);
      const imageHash = await this.calculateImageHash(imageFile);
      
      return {
        features,
        imageHash
      };
    }
  }

  private createFallbackFeatures(canvas: HTMLCanvasElement): number[] {
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Create features from image statistics and patterns
    const features: number[] = [];
    
    // Grid-based feature extraction (similar to deep learning approach)
    const gridSize = 8;
    const cellSize = canvas.width / gridSize;
    
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        let rSum = 0, gSum = 0, bSum = 0;
        let pixelCount = 0;
        
        // Calculate average color for each grid cell
        for (let dy = 0; dy < cellSize; dy++) {
          for (let dx = 0; dx < cellSize; dx++) {
            const px = Math.floor(x * cellSize + dx);
            const py = Math.floor(y * cellSize + dy);
            
            if (px < canvas.width && py < canvas.height) {
              const idx = (py * canvas.width + px) * 4;
              rSum += data[idx];
              gSum += data[idx + 1];
              bSum += data[idx + 2];
              pixelCount++;
            }
          }
        }
        
        if (pixelCount > 0) {
          features.push(rSum / pixelCount / 255);
          features.push(gSum / pixelCount / 255);
          features.push(bSum / pixelCount / 255);
        }
      }
    }
    
    // Add edge detection features
    const edgeFeatures = this.calculateEdgeFeatures(imageData);
    features.push(...edgeFeatures);
    
    // Normalize features
    const norm = Math.sqrt(features.reduce((sum, val) => sum + val * val, 0));
    return features.map(val => val / (norm || 1));
  }

  private calculateEdgeFeatures(imageData: ImageData): number[] {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const edges: number[] = [];
    
    // Simple edge detection using Sobel-like operator
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        
        // Convert to grayscale
        const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        
        // Calculate gradient
        const gx = (data[((y - 1) * width + (x + 1)) * 4] - data[((y - 1) * width + (x - 1)) * 4]) +
                   2 * (data[(y * width + (x + 1)) * 4] - data[(y * width + (x - 1)) * 4]) +
                   (data[((y + 1) * width + (x + 1)) * 4] - data[((y + 1) * width + (x - 1)) * 4]);
        
        const gy = (data[((y - 1) * width + (x - 1)) * 4] + 2 * data[((y - 1) * width + x) * 4] + data[((y - 1) * width + (x + 1)) * 4]) -
                   (data[((y + 1) * width + (x - 1)) * 4] + 2 * data[((y + 1) * width + x) * 4] + data[((y + 1) * width + (x + 1)) * 4]);
        
        const magnitude = Math.sqrt(gx * gx + gy * gy);
        edges.push(magnitude / 255);
      }
    }
    
    // Return averaged edge features
    const blockSize = 16;
    const edgeFeatures: number[] = [];
    for (let i = 0; i < edges.length; i += blockSize) {
      const block = edges.slice(i, i + blockSize);
      const avg = block.reduce((sum, val) => sum + val, 0) / block.length;
      edgeFeatures.push(avg);
    }
    
    return edgeFeatures;
  }

  calculateCosineSimilarity(features1: number[], features2: number[]): number {
    if (features1.length !== features2.length) {
      console.warn('Feature vectors have different lengths');
      return 0;
    }

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < features1.length; i++) {
      dotProduct += features1[i] * features2[i];
      norm1 += features1[i] * features1[i];
      norm2 += features2[i] * features2[i];
    }

    const denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
    return denominator === 0 ? 0 : dotProduct / denominator;
  }

  async verifyPalm(paymentImage: File): Promise<{
    isMatch: boolean;
    similarity: number;
    userInfo?: EnrollmentData;
    finalAmount: number;
  }> {
    try {
      console.log('Starting palm verification...');
      
      // Extract features from payment image
      const paymentFeatures = await this.extractFeatures(paymentImage);
      
      // Get all enrolled palms from localStorage
      const enrolledData: EnrollmentData[] = JSON.parse(localStorage.getItem('palmEnrollments') || '[]');
      
      if (enrolledData.length === 0) {
        console.log('No enrolled palms found');
        return {
          isMatch: false,
          similarity: 0,
          finalAmount: 30
        };
      }

      let bestMatch: EnrollmentData | null = null;
      let bestSimilarity = 0;

      // Perform 1:N matching (like Amazon One)
      for (const enrollment of enrolledData) {
        if (!enrollment.palmFeatures || !enrollment.palmFeatures.features) {
          console.warn('Invalid enrollment data, skipping');
          continue;
        }

        const similarity = this.calculateCosineSimilarity(
          paymentFeatures.features,
          enrollment.palmFeatures.features
        );

        console.log(`Similarity with ${enrollment.fullName}: ${similarity.toFixed(3)}`);

        if (similarity > bestSimilarity) {
          bestSimilarity = similarity;
          bestMatch = enrollment;
        }
      }

      const isMatch = bestSimilarity >= this.SIMILARITY_THRESHOLD;
      const originalAmount = 30;
      let finalAmount = originalAmount;

      if (isMatch && bestMatch) {
        const discountAmount = parseFloat(bestMatch.moneyAmount) || 0;
        finalAmount = Math.max(0, originalAmount - discountAmount);
        
        console.log(`✅ Palm verified! User: ${bestMatch.fullName}, Similarity: ${bestSimilarity.toFixed(3)}`);
        console.log(`Original: $${originalAmount}, Discount: $${discountAmount}, Final: $${finalAmount}`);
      } else {
        console.log(`❌ Palm verification failed. Best similarity: ${bestSimilarity.toFixed(3)} (threshold: ${this.SIMILARITY_THRESHOLD})`);
      }

      return {
        isMatch,
        similarity: bestSimilarity,
        userInfo: isMatch ? bestMatch || undefined : undefined,
        finalAmount
      };

    } catch (error) {
      console.error('Error during palm verification:', error);
      return {
        isMatch: false,
        similarity: 0,
        finalAmount: 30
      };
    }
  }
}

// Export singleton instance
export const palmFeatureExtractor = new PalmFeatureExtractor();