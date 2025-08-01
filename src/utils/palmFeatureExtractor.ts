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
  private readonly SIMILARITY_THRESHOLD = 0.65;

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
        
        // Draw image
        ctx.drawImage(img, 0, 0, 224, 224);
        
        // Advanced preprocessing for palm recognition
        const imageData = ctx.getImageData(0, 0, 224, 224);
        const data = imageData.data;
        
        // Convert to grayscale and normalize
        const grayData = new Uint8Array(224 * 224);
        for (let i = 0; i < data.length; i += 4) {
          const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
          grayData[i / 4] = gray;
        }
        
        // Apply histogram equalization for better contrast
        const histogram = new Array(256).fill(0);
        for (let i = 0; i < grayData.length; i++) {
          histogram[grayData[i]]++;
        }
        
        // Calculate cumulative distribution
        const cdf = new Array(256);
        cdf[0] = histogram[0];
        for (let i = 1; i < 256; i++) {
          cdf[i] = cdf[i - 1] + histogram[i];
        }
        
        // Normalize CDF
        const totalPixels = grayData.length;
        for (let i = 0; i < 256; i++) {
          cdf[i] = Math.round((cdf[i] / totalPixels) * 255);
        }
        
        // Apply equalization and enhance contrast
        for (let i = 0; i < grayData.length; i++) {
          const equalizedValue = cdf[grayData[i]];
          // Enhanced contrast with sigmoid-like function
          const enhanced = Math.round(255 / (1 + Math.exp(-0.05 * (equalizedValue - 128))));
          grayData[i] = enhanced;
        }
        
        // Apply Gaussian blur to reduce noise
        const blurred = this.gaussianBlur(grayData, 224, 224, 1.0);
        
        // Edge enhancement using unsharp masking
        const sharpened = this.unsharpMask(blurred, grayData, 224, 224, 0.5);
        
        // Convert back to RGB
        for (let i = 0; i < sharpened.length; i++) {
          const value = sharpened[i];
          data[i * 4] = value;     // R
          data[i * 4 + 1] = value; // G
          data[i * 4 + 2] = value; // B
          // Alpha channel remains unchanged
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

  private gaussianBlur(data: Uint8Array, width: number, height: number, sigma: number): Uint8Array {
    const result = new Uint8Array(data.length);
    const kernelSize = Math.ceil(sigma * 3) * 2 + 1;
    const kernel = this.createGaussianKernel(kernelSize, sigma);
    
    // Horizontal pass
    const temp = new Uint8Array(data.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0;
        let weightSum = 0;
        
        for (let k = 0; k < kernelSize; k++) {
          const px = x + k - Math.floor(kernelSize / 2);
          if (px >= 0 && px < width) {
            sum += data[y * width + px] * kernel[k];
            weightSum += kernel[k];
          }
        }
        temp[y * width + x] = Math.round(sum / weightSum);
      }
    }
    
    // Vertical pass
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0;
        let weightSum = 0;
        
        for (let k = 0; k < kernelSize; k++) {
          const py = y + k - Math.floor(kernelSize / 2);
          if (py >= 0 && py < height) {
            sum += temp[py * width + x] * kernel[k];
            weightSum += kernel[k];
          }
        }
        result[y * width + x] = Math.round(sum / weightSum);
      }
    }
    
    return result;
  }

  private createGaussianKernel(size: number, sigma: number): number[] {
    const kernel = new Array(size);
    const center = Math.floor(size / 2);
    let sum = 0;
    
    for (let i = 0; i < size; i++) {
      const x = i - center;
      kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
      sum += kernel[i];
    }
    
    // Normalize
    for (let i = 0; i < size; i++) {
      kernel[i] /= sum;
    }
    
    return kernel;
  }

  private unsharpMask(blurred: Uint8Array, original: Uint8Array, width: number, height: number, amount: number): Uint8Array {
    const result = new Uint8Array(original.length);
    
    for (let i = 0; i < original.length; i++) {
      const mask = original[i] - blurred[i];
      const sharpened = original[i] + amount * mask;
      result[i] = Math.max(0, Math.min(255, Math.round(sharpened)));
    }
    
    return result;
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
    
    // Convert to grayscale for better analysis
    const grayData = new Uint8Array(canvas.width * canvas.height);
    for (let i = 0; i < data.length; i += 4) {
      const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
      grayData[i / 4] = gray;
    }
    
    // Multi-scale grid features for different palm structures
    const gridSizes = [4, 8, 16]; // Different scales to capture various palm features
    
    for (const gridSize of gridSizes) {
      const cellSize = canvas.width / gridSize;
      
      for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
          let intensitySum = 0;
          let varianceSum = 0;
          let pixelCount = 0;
          const cellPixels: number[] = [];
          
          // Collect pixels in this cell
          for (let dy = 0; dy < cellSize; dy++) {
            for (let dx = 0; dx < cellSize; dx++) {
              const px = Math.floor(x * cellSize + dx);
              const py = Math.floor(y * cellSize + dy);
              
              if (px < canvas.width && py < canvas.height) {
                const grayValue = grayData[py * canvas.width + px];
                cellPixels.push(grayValue);
                intensitySum += grayValue;
                pixelCount++;
              }
            }
          }
          
          if (pixelCount > 0) {
            // Mean intensity
            const meanIntensity = intensitySum / pixelCount;
            features.push(meanIntensity / 255);
            
            // Variance (texture information)
            for (const pixel of cellPixels) {
              varianceSum += Math.pow(pixel - meanIntensity, 2);
            }
            const variance = varianceSum / pixelCount;
            features.push(Math.sqrt(variance) / 255);
            
            // Entropy (randomness measure - important for palm texture)
            const histogram = new Array(256).fill(0);
            for (const pixel of cellPixels) {
              histogram[pixel]++;
            }
            let entropy = 0;
            for (let i = 0; i < 256; i++) {
              if (histogram[i] > 0) {
                const probability = histogram[i] / pixelCount;
                entropy -= probability * Math.log2(probability);
              }
            }
            features.push(entropy / 8); // Normalize entropy
          }
        }
      }
    }
    
    // Add directional edge features (palm lines have specific orientations)
    const directionalEdges = this.calculateDirectionalEdges(grayData, canvas.width, canvas.height);
    features.push(...directionalEdges);
    
    // Add texture features using Local Binary Patterns
    const lbpFeatures = this.calculateLBPFeatures(grayData, canvas.width, canvas.height);
    features.push(...lbpFeatures);
    
    // Normalize features using robust scaling
    const robustFeatures = this.robustNormalization(features);
    
    return robustFeatures;
  }

  private calculateDirectionalEdges(grayData: Uint8Array, width: number, height: number): number[] {
    const features: number[] = [];
    
    // Sobel operators for different directions
    const sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    const sobelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
    const diagonal1 = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]];
    const diagonal2 = [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]];
    
    const operators = [sobelX, sobelY, diagonal1, diagonal2];
    
    for (const operator of operators) {
      let edgeSum = 0;
      let count = 0;
      
      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          let gradient = 0;
          
          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              const pixel = grayData[(y + ky) * width + (x + kx)];
              gradient += pixel * operator[ky + 1][kx + 1];
            }
          }
          
          edgeSum += Math.abs(gradient);
          count++;
        }
      }
      
      features.push(edgeSum / count / 1020); // Normalize
    }
    
    return features;
  }

  private calculateLBPFeatures(grayData: Uint8Array, width: number, height: number): number[] {
    const lbpHistogram = new Array(256).fill(0);
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const center = grayData[y * width + x];
        let lbpValue = 0;
        
        // 8-connected neighbors
        const neighbors = [
          grayData[(y - 1) * width + (x - 1)], // Top-left
          grayData[(y - 1) * width + x],       // Top
          grayData[(y - 1) * width + (x + 1)], // Top-right
          grayData[y * width + (x + 1)],       // Right
          grayData[(y + 1) * width + (x + 1)], // Bottom-right
          grayData[(y + 1) * width + x],       // Bottom
          grayData[(y + 1) * width + (x - 1)], // Bottom-left
          grayData[y * width + (x - 1)]        // Left
        ];
        
        for (let i = 0; i < 8; i++) {
          if (neighbors[i] >= center) {
            lbpValue |= (1 << i);
          }
        }
        
        lbpHistogram[lbpValue]++;
      }
    }
    
    // Normalize histogram
    const totalPixels = (width - 2) * (height - 2);
    return lbpHistogram.map(count => count / totalPixels);
  }

  private robustNormalization(features: number[]): number[] {
    if (features.length === 0) return features;
    
    // Calculate median and MAD (Median Absolute Deviation) for robust scaling
    const sortedFeatures = [...features].sort((a, b) => a - b);
    const median = sortedFeatures[Math.floor(sortedFeatures.length / 2)];
    
    const deviations = features.map(f => Math.abs(f - median));
    const sortedDeviations = deviations.sort((a, b) => a - b);
    const mad = sortedDeviations[Math.floor(sortedDeviations.length / 2)];
    
    // Robust scaling: (x - median) / MAD
    const scalingFactor = mad > 0 ? mad : 1;
    const scaledFeatures = features.map(f => (f - median) / scalingFactor);
    
    // Final L2 normalization
    const norm = Math.sqrt(scaledFeatures.reduce((sum, val) => sum + val * val, 0));
    return scaledFeatures.map(val => val / (norm || 1));
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
      console.log('Model initialized:', !!this.model);
      
      // Ensure model is initialized
      if (!this.model) {
        console.log('Model not initialized, initializing now...');
        await this.initialize();
      }
      
      // Extract features from payment image
      const paymentFeatures = await this.extractFeatures(paymentImage);
      console.log('Payment features extracted, length:', paymentFeatures.features.length);
      console.log('Feature sample (first 5):', paymentFeatures.features.slice(0, 5));
      
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

        console.log(`Enrolled features length: ${enrollment.palmFeatures.features.length}`);
        console.log(`Enrolled feature sample (first 5):`, enrollment.palmFeatures.features.slice(0, 5));

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

      console.log(`Best similarity: ${bestSimilarity}, Threshold: ${this.SIMILARITY_THRESHOLD}`);
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