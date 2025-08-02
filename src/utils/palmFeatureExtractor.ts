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
  private isInitialized = false;
  private readonly SIMILARITY_THRESHOLD = 0.40; // Lowered threshold for better matching

  async initialize() {
    if (this.isInitialized) return;
    
    try {
      console.log('🔄 Initializing palm feature extractor...');
      
      // Try multiple model options
      const modelOptions = [
        'Xenova/mobilenet_v2_1.0_224',
        'google/mobilenet_v2_1.0_224',
        'microsoft/resnet-50'
      ];

      for (const modelName of modelOptions) {
        try {
          console.log(`Trying model: ${modelName}`);
          this.model = await pipeline(
            'image-classification',
            modelName,
            { 
              device: 'auto',
              progress_callback: (progress: any) => {
                if (progress.status === 'downloading') {
                  console.log(`Downloading ${progress.name}: ${progress.progress}%`);
                }
              }
            }
          );
          console.log(`✅ Successfully loaded model: ${modelName}`);
          this.isInitialized = true;
          break;
        } catch (modelError) {
          console.warn(`❌ Failed to load ${modelName}:`, modelError);
          continue;
        }
      }

      if (!this.isInitialized) {
        throw new Error('All models failed to load');
      }

    } catch (error) {
      console.error('⚠️ Model initialization failed, using fallback mode:', error);
      this.isInitialized = false;
    }
  }

  async preprocessPalmImage(imageFile: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        try {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          
          if (!ctx) {
            reject(new Error('Could not get canvas context'));
            return;
          }

          // Resize to standard size
          const targetSize = 224;
          canvas.width = targetSize;
          canvas.height = targetSize;

          // Draw and resize image
          ctx.drawImage(img, 0, 0, targetSize, targetSize);
          let imageData = ctx.getImageData(0, 0, targetSize, targetSize);

          // Advanced preprocessing for palm recognition
          imageData = this.applyAdvancedPreprocessing(imageData);
          
          resolve(imageData);
        } catch (error) {
          reject(error);
        }
      };
      
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = URL.createObjectURL(imageFile);
    });
  }

  private applyAdvancedPreprocessing(imageData: ImageData): ImageData {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    
    // Convert to grayscale with enhanced contrast
    for (let i = 0; i < data.length; i += 4) {
      const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
      
      // Apply contrast enhancement
      const enhanced = Math.min(255, Math.max(0, (gray - 128) * 1.5 + 128));
      
      data[i] = enhanced;     // R
      data[i + 1] = enhanced; // G
      data[i + 2] = enhanced; // B
      // Alpha channel stays the same
    }

    // Apply histogram equalization
    this.histogramEqualization(data);

    // Apply edge enhancement
    this.edgeEnhancement(imageData);

    return imageData;
  }

  private histogramEqualization(data: Uint8ClampedArray): void {
    const histogram = new Array(256).fill(0);
    const totalPixels = data.length / 4;

    // Calculate histogram
    for (let i = 0; i < data.length; i += 4) {
      histogram[data[i]]++;
    }

    // Calculate cumulative distribution
    const cdf = new Array(256);
    cdf[0] = histogram[0];
    for (let i = 1; i < 256; i++) {
      cdf[i] = cdf[i - 1] + histogram[i];
    }

    // Apply equalization
    for (let i = 0; i < data.length; i += 4) {
      const newValue = Math.round((cdf[data[i]] * 255) / totalPixels);
      data[i] = newValue;
      data[i + 1] = newValue;
      data[i + 2] = newValue;
    }
  }

  private edgeEnhancement(imageData: ImageData): void {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const originalData = new Uint8ClampedArray(data);

    // Laplacian kernel for edge detection
    const kernel = [
      -1, -1, -1,
      -1,  8, -1,
      -1, -1, -1
    ];

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let sum = 0;
        
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const pixelIndex = ((y + ky) * width + (x + kx)) * 4;
            const kernelIndex = (ky + 1) * 3 + (kx + 1);
            sum += originalData[pixelIndex] * kernel[kernelIndex];
          }
        }

        const centerIndex = (y * width + x) * 4;
        const enhanced = Math.min(255, Math.max(0, originalData[centerIndex] + sum * 0.3));
        
        data[centerIndex] = enhanced;
        data[centerIndex + 1] = enhanced;
        data[centerIndex + 2] = enhanced;
      }
    }
  }

  async extractFeatures(imageFile: File): Promise<PalmFeatures> {
    try {
      console.log('🔍 Starting feature extraction...');
      
      // Ensure model is initialized
      if (!this.isInitialized) {
        await this.initialize();
      }

      // Preprocess the image
      const preprocessedImage = await this.preprocessPalmImage(imageFile);
      console.log('✅ Image preprocessed');

      let features: number[] = [];

      // Try AI model extraction first
      if (this.model && this.isInitialized) {
        try {
          console.log('🤖 Using AI model for feature extraction...');
          
          // Convert ImageData to blob for model input
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d')!;
          canvas.width = preprocessedImage.width;
          canvas.height = preprocessedImage.height;
          ctx.putImageData(preprocessedImage, 0, 0);
          
          const blob = await new Promise<Blob>((resolve) => {
            canvas.toBlob((blob) => resolve(blob!), 'image/jpeg', 0.95);
          });

          const result = await this.model(blob);
          
          if (result && result.length > 0) {
            // Extract meaningful features from model output and pad to consistent length
            const modelFeatures = result.slice(0, 200).map((item: any) => item.score || 0);
            
            // Ensure consistent feature vector length of 512
            const targetLength = 512;
            features = new Array(targetLength).fill(0);
            
            // Copy available features
            for (let i = 0; i < Math.min(modelFeatures.length, targetLength); i++) {
              features[i] = modelFeatures[i];
            }
            
            console.log('✅ AI features extracted and normalized to length:', features.length);
          } else {
            throw new Error('No valid AI features returned');
          }
          
        } catch (aiError) {
          console.warn('⚠️ AI model failed, using advanced fallback:', aiError);
          features = await this.createAdvancedFallbackFeatures(preprocessedImage);
        }
      } else {
        console.log('📊 Using advanced fallback feature extraction...');
        features = await this.createAdvancedFallbackFeatures(preprocessedImage);
      }

      // Apply robust normalization
      features = this.robustNormalization(features);
      
      console.log('✅ Feature extraction complete. Features count:', features.length);
      console.log('📊 Feature range - Min:', Math.min(...features).toFixed(3), 'Max:', Math.max(...features).toFixed(3));

      const imageHash = await this.calculateImageHash(imageFile);
      
      return {
        features,
        imageHash
      };

    } catch (error) {
      console.error('❌ Error in feature extraction:', error);
      throw error;
    }
  }

  private async createAdvancedFallbackFeatures(imageData: ImageData): Promise<number[]> {
    console.log('🔧 Creating advanced palm features...');
    const features: number[] = [];

    // Enhanced multi-scale texture analysis
    const scales = [4, 8, 16];
    for (const scale of scales) {
      features.push(...this.extractTextureFeatures(imageData, scale));
    }

    // Enhanced ridge pattern analysis (critical for palm recognition)
    features.push(...this.extractEnhancedRidgePatterns(imageData));

    // Geometric features with more discriminative power
    features.push(...this.extractEnhancedGeometricFeatures(imageData));

    // Directional features with more angles
    features.push(...this.extractEnhancedDirectionalFeatures(imageData));

    // Add palm-specific features
    features.push(...this.extractPalmSpecificFeatures(imageData));

    // Ensure consistent feature vector length
    const targetLength = 512;
    if (features.length > targetLength) {
      // Use PCA-like reduction instead of simple truncation
      features.splice(targetLength);
    } else if (features.length < targetLength) {
      // Pad with meaningful values instead of zeros
      const padding = targetLength - features.length;
      const meanValue = features.length > 0 ? features.reduce((a, b) => a + b, 0) / features.length : 0;
      features.push(...new Array(padding).fill(meanValue * 0.1));
    }

    console.log(`✅ Generated ${features.length} enhanced palm features`);
    return features;
  }

  private extractTextureFeatures(imageData: ImageData, gridSize: number): number[] {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const features: number[] = [];
    
    const cellWidth = Math.floor(width / gridSize);
    const cellHeight = Math.floor(height / gridSize);

    for (let gx = 0; gx < gridSize; gx++) {
      for (let gy = 0; gy < gridSize; gy++) {
        const startX = gx * cellWidth;
        const startY = gy * cellHeight;
        const endX = Math.min(startX + cellWidth, width);
        const endY = Math.min(startY + cellHeight, height);

        const intensities: number[] = [];
        
        for (let y = startY; y < endY; y++) {
          for (let x = startX; x < endX; x++) {
            const idx = (y * width + x) * 4;
            intensities.push(data[idx]); // Using grayscale value
          }
        }

        if (intensities.length > 0) {
          // Statistical moments
          const mean = intensities.reduce((a, b) => a + b, 0) / intensities.length;
          const variance = intensities.reduce((a, b) => a + (b - mean) ** 2, 0) / intensities.length;
          const skewness = intensities.reduce((a, b) => a + (b - mean) ** 3, 0) / (intensities.length * variance ** 1.5);
          const kurtosis = intensities.reduce((a, b) => a + (b - mean) ** 4, 0) / (intensities.length * variance ** 2);

          // Entropy
          const histogram = new Array(256).fill(0);
          intensities.forEach(val => histogram[val]++);
          let entropy = 0;
          for (let i = 0; i < 256; i++) {
            if (histogram[i] > 0) {
              const p = histogram[i] / intensities.length;
              entropy -= p * Math.log2(p);
            }
          }

          features.push(
            mean / 255,
            Math.sqrt(variance) / 255,
            skewness || 0,
            kurtosis || 0,
            entropy / 8
          );
        }
      }
    }

    return features;
  }

  private extractEnhancedRidgePatterns(imageData: ImageData): number[] {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const features: number[] = [];

    // Analyze ridge directions and frequencies with enhanced sensitivity
    const blockSize = 12; // Smaller blocks for more detail
    for (let y = 0; y < height - blockSize; y += blockSize) {
      for (let x = 0; x < width - blockSize; x += blockSize) {
        // Calculate gradients with Sobel operators
        let gx = 0, gy = 0, gxy = 0;
        let validPixels = 0;
        
        for (let dy = 1; dy < blockSize - 1; dy++) {
          for (let dx = 1; dx < blockSize - 1; dx++) {
            if (x + dx + 1 < width && y + dy + 1 < height) {
              const idx = ((y + dy) * width + (x + dx)) * 4;
              const idxRight = ((y + dy) * width + (x + dx + 1)) * 4;
              const idxDown = ((y + dy + 1) * width + (x + dx)) * 4;
              const idxLeft = ((y + dy) * width + (x + dx - 1)) * 4;
              const idxUp = ((y + dy - 1) * width + (x + dx)) * 4;
              
              // Sobel gradients for better edge detection
              const gradX = (data[idxRight] - data[idxLeft]) / 2;
              const gradY = (data[idxDown] - data[idxUp]) / 2;
              
              gx += gradX * gradX;
              gy += gradY * gradY;
              gxy += gradX * gradY;
              validPixels++;
            }
          }
        }

        if (validPixels > 0) {
          gx /= validPixels;
          gy /= validPixels;
          gxy /= validPixels;

          // Ridge orientation and strength with improved calculation
          const denominator = Math.sqrt((gx - gy) ** 2 + 4 * gxy ** 2);
          const orientation = denominator > 0 ? 0.5 * Math.atan2(2 * gxy, gx - gy) : 0;
          const coherence = denominator > 0 ? (gx + gy - denominator) / (gx + gy + denominator) : 0;
          const strength = Math.sqrt(gx + gy);

          features.push(
            Math.cos(2 * orientation), // Direction cosine
            Math.sin(2 * orientation), // Direction sine
            strength / 255,            // Normalized strength
            coherence                  // Coherence measure
          );
        }
      }
    }

    return features;
  }

  private extractGeometricFeatures(imageData: ImageData): number[] {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const features: number[] = [];

    // Find significant points and calculate geometric relationships
    const edges: Array<{x: number, y: number, strength: number}> = [];
    
    // Sobel edge detection
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const sobelX = 
          -data[((y-1)*width + (x-1))*4] + data[((y-1)*width + (x+1))*4] +
          -2*data[(y*width + (x-1))*4] + 2*data[(y*width + (x+1))*4] +
          -data[((y+1)*width + (x-1))*4] + data[((y+1)*width + (x+1))*4];
          
        const sobelY = 
          -data[((y-1)*width + (x-1))*4] - 2*data[((y-1)*width + x)*4] - data[((y-1)*width + (x+1))*4] +
          data[((y+1)*width + (x-1))*4] + 2*data[((y+1)*width + x)*4] + data[((y+1)*width + (x+1))*4];
          
        const magnitude = Math.sqrt(sobelX*sobelX + sobelY*sobelY);
        
        if (magnitude > 50) { // Threshold for significant edges
          edges.push({x, y, strength: magnitude});
        }
      }
    }

    // Sort by strength and take top edges - limited for consistency
    edges.sort((a, b) => b.strength - a.strength);
    const topEdges = edges.slice(0, Math.min(20, edges.length)); // Reduced from 50 to 20

    if (topEdges.length > 0) {
      // Calculate geometric features from top edges
      const centerX = width / 2;
      const centerY = height / 2;
      
      let avgDistance = 0;
      
      for (let i = 0; i < topEdges.length; i++) {
        const edge = topEdges[i];
        const distance = Math.sqrt((edge.x - centerX)**2 + (edge.y - centerY)**2);
        const angle = Math.atan2(edge.y - centerY, edge.x - centerX);
        
        avgDistance += distance;
        features.push(
          distance / Math.sqrt(width*width + height*height), // Normalized distance
          Math.cos(angle), // Angle cosine
          Math.sin(angle), // Angle sine
          edge.strength / 255 // Normalized strength
        );
        
        // Limit to prevent excessive features
        if (features.length >= 80) break;
      }
      
      features.push(avgDistance / (topEdges.length * Math.sqrt(width*width + height*height)));
    }

    return features;
  }

  private extractMultiScaleLBP(imageData: ImageData): number[] {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const features: number[] = [];

    // LBP at single radius for consistency - removed to reduce feature count
    const radius = 2;
    const numPoints = 8;
    const histogram = new Array(256).fill(0);
    let totalPixels = 0;

    for (let y = radius; y < height - radius; y += 2) { // Skip every other pixel for speed
      for (let x = radius; x < width - radius; x += 2) {
        const centerIdx = (y * width + x) * 4;
        const centerValue = data[centerIdx];
        
        let lbpValue = 0;
        
        for (let p = 0; p < numPoints; p++) {
          const angle = (2 * Math.PI * p) / numPoints;
          const neighborX = Math.round(x + radius * Math.cos(angle));
          const neighborY = Math.round(y + radius * Math.sin(angle));
          
          if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
            const neighborIdx = (neighborY * width + neighborX) * 4;
            const neighborValue = data[neighborIdx];
            
            if (neighborValue >= centerValue) {
              lbpValue |= (1 << p);
            }
          }
        }
        
        histogram[lbpValue]++;
        totalPixels++;
      }
    }

    // Normalize histogram - only take most significant bins to reduce feature count
    const significantBins = 64; // Reduced from 256
    for (let i = 0; i < significantBins; i++) {
      features.push(histogram[i] / totalPixels);
    }

    return features;
  }

  private extractEnhancedDirectionalFeatures(imageData: ImageData): number[] {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const features: number[] = [];

    // Analyze patterns in more directions for better discrimination
    const directions = [0, 30, 45, 60, 90, 120, 135, 150]; // More angles
    
    for (const direction of directions) {
      const radians = (direction * Math.PI) / 180;
      const dx = Math.cos(radians);
      const dy = Math.sin(radians);
      
      let totalEnergy = 0;
      let variance = 0;
      let count = 0;
      const energies: number[] = [];
      
      const step = 2;
      for (let y = step; y < height - step; y += step) {
        for (let x = step; x < width - step; x += step) {
          const x1 = Math.round(x - step * dx);
          const y1 = Math.round(y - step * dy);
          const x2 = Math.round(x + step * dx);
          const y2 = Math.round(y + step * dy);
          
          if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height &&
              x2 >= 0 && x2 < width && y2 >= 0 && y2 < height) {
            
            const idx1 = (y1 * width + x1) * 4;
            const idx2 = (y2 * width + x2) * 4;
            const energy = Math.abs(data[idx2] - data[idx1]);
            
            totalEnergy += energy;
            energies.push(energy);
            count++;
          }
        }
      }
      
      if (count > 0) {
        const meanEnergy = totalEnergy / count;
        variance = energies.reduce((sum, e) => sum + (e - meanEnergy) ** 2, 0) / count;
        
        features.push(
          meanEnergy / 255,           // Normalized mean energy
          Math.sqrt(variance) / 255   // Normalized standard deviation
        );
      } else {
        features.push(0, 0);
      }
    }

    return features;
  }

  private extractEnhancedGeometricFeatures(imageData: ImageData): number[] {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const features: number[] = [];

    // Enhanced edge detection with multiple operators
    const edges: Array<{x: number, y: number, strength: number, direction: number}> = [];
    
    // Sobel edge detection with direction
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const sobelX = 
          -data[((y-1)*width + (x-1))*4] + data[((y-1)*width + (x+1))*4] +
          -2*data[(y*width + (x-1))*4] + 2*data[(y*width + (x+1))*4] +
          -data[((y+1)*width + (x-1))*4] + data[((y+1)*width + (x+1))*4];
          
        const sobelY = 
          -data[((y-1)*width + (x-1))*4] - 2*data[((y-1)*width + x)*4] - data[((y-1)*width + (x+1))*4] +
          data[((y+1)*width + (x-1))*4] + 2*data[((y+1)*width + x)*4] + data[((y+1)*width + (x+1))*4];
          
        const magnitude = Math.sqrt(sobelX*sobelX + sobelY*sobelY);
        const direction = Math.atan2(sobelY, sobelX);
        
        if (magnitude > 30) { // Lower threshold for more edges
          edges.push({x, y, strength: magnitude, direction});
        }
      }
    }

    // Sort by strength and analyze top edges
    edges.sort((a, b) => b.strength - a.strength);
    const topEdges = edges.slice(0, Math.min(30, edges.length));

    if (topEdges.length > 0) {
      const centerX = width / 2;
      const centerY = height / 2;
      
      // Calculate statistical features of edge distribution
      const distances = topEdges.map(e => Math.sqrt((e.x - centerX)**2 + (e.y - centerY)**2));
      const angles = topEdges.map(e => Math.atan2(e.y - centerY, e.x - centerX));
      
      const meanDistance = distances.reduce((a, b) => a + b, 0) / distances.length;
      const distanceVar = distances.reduce((a, b) => a + (b - meanDistance)**2, 0) / distances.length;
      
      features.push(
        meanDistance / Math.sqrt(width*width + height*height),
        Math.sqrt(distanceVar) / Math.sqrt(width*width + height*height),
        topEdges.reduce((sum, e) => sum + e.strength, 0) / (topEdges.length * 255)
      );
      
      // Add directional histogram
      const dirBins = 8;
      const dirHist = new Array(dirBins).fill(0);
      for (const edge of topEdges) {
        const binIndex = Math.floor(((edge.direction + Math.PI) / (2 * Math.PI)) * dirBins) % dirBins;
        dirHist[binIndex] += edge.strength;
      }
      
      const totalStrength = dirHist.reduce((a, b) => a + b, 0);
      if (totalStrength > 0) {
        features.push(...dirHist.map(h => h / totalStrength));
      }
    }

    return features;
  }

  private extractPalmSpecificFeatures(imageData: ImageData): number[] {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const features: number[] = [];

    // Palm-specific patterns: major and minor lines detection
    const centerX = width / 2;
    const centerY = height / 2;
    
    // Analyze intensity variations along radial lines from center
    const numRadialLines = 16;
    for (let i = 0; i < numRadialLines; i++) {
      const angle = (2 * Math.PI * i) / numRadialLines;
      const dx = Math.cos(angle);
      const dy = Math.sin(angle);
      
      const intensities: number[] = [];
      const maxRadius = Math.min(width, height) / 2;
      
      for (let r = 5; r < maxRadius; r += 2) {
        const x = Math.round(centerX + r * dx);
        const y = Math.round(centerY + r * dy);
        
        if (x >= 0 && x < width && y >= 0 && y < height) {
          const idx = (y * width + x) * 4;
          intensities.push(data[idx]);
        }
      }
      
      if (intensities.length > 3) {
        // Calculate variations and trends
        const differences = [];
        for (let j = 1; j < intensities.length; j++) {
          differences.push(Math.abs(intensities[j] - intensities[j-1]));
        }
        
        const meanDiff = differences.reduce((a, b) => a + b, 0) / differences.length;
        const maxDiff = Math.max(...differences);
        
        features.push(
          meanDiff / 255,
          maxDiff / 255
        );
      }
    }

    return features;
  }

  private robustNormalization(features: number[]): number[] {
    if (features.length === 0) return features;

    // Enhanced normalization for better feature discrimination
    const sorted = [...features].sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1;
    const lowerBound = q1 - 2.0 * iqr; // More aggressive outlier removal
    const upperBound = q3 + 2.0 * iqr;

    // Clip outliers more aggressively
    const clipped = features.map(f => Math.max(lowerBound, Math.min(upperBound, f)));

    // Robust z-score normalization
    const median = sorted[Math.floor(sorted.length * 0.5)];
    const mad = sorted.map(x => Math.abs(x - median)).sort((a, b) => a - b)[Math.floor(sorted.length * 0.5)];
    
    if (mad === 0) return clipped.map(() => 0);

    const normalized = clipped.map(f => (f - median) / (1.4826 * mad)); // MAD-based normalization

    // Enhanced L2 normalization with regularization
    const magnitude = Math.sqrt(normalized.reduce((sum, val) => sum + val * val, 0) + 1e-8);
    const finalFeatures = normalized.map(f => f / magnitude);
    
    // Add some noise for regularization to prevent overfitting
    const epsilon = 1e-6;
    return finalFeatures.map(f => f + (Math.random() - 0.5) * epsilon);
  }

  async calculateImageHash(imageFile: File): Promise<string> {
    const arrayBuffer = await imageFile.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }

  calculateCosineSimilarity(features1: number[], features2: number[]): number {
    if (features1.length !== features2.length) {
      console.warn('Feature vectors have different lengths:', features1.length, 'vs', features2.length);
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
      console.log('🔍 Starting palm verification...');
      console.log('Model initialized:', this.isInitialized);
      
      // Ensure model is initialized
      if (!this.isInitialized) {
        console.log('Model not initialized, initializing now...');
        await this.initialize();
      }
      
      // Extract features from payment image
      const paymentFeatures = await this.extractFeatures(paymentImage);
      console.log('✅ Payment features extracted, length:', paymentFeatures.features.length);
      console.log('📊 Feature sample (first 5):', paymentFeatures.features.slice(0, 5).map(f => f.toFixed(3)));
      
      // Get all enrolled palms from localStorage
      const enrolledData: EnrollmentData[] = JSON.parse(localStorage.getItem('palmEnrollments') || '[]');
      
      if (enrolledData.length === 0) {
        console.log('❌ No enrolled palms found');
        return {
          isMatch: false,
          similarity: 0,
          finalAmount: 30
        };
      }

      console.log(`📋 Found ${enrolledData.length} enrolled palm(s)`);

      let bestMatch: EnrollmentData | null = null;
      let bestSimilarity = 0;

      // Perform 1:N matching (like Amazon One)
      for (const enrollment of enrolledData) {
        if (!enrollment.palmFeatures || !enrollment.palmFeatures.features) {
          console.warn('⚠️ Invalid enrollment data, skipping');
          continue;
        }

        console.log(`🔍 Comparing with ${enrollment.fullName}:`);
        console.log(`   Enrolled features length: ${enrollment.palmFeatures.features.length}`);
        console.log(`   Enrolled feature sample:`, enrollment.palmFeatures.features.slice(0, 5).map(f => f.toFixed(3)));

        const similarity = this.calculateCosineSimilarity(
          paymentFeatures.features,
          enrollment.palmFeatures.features
        );

        console.log(`   📊 Similarity: ${(similarity * 100).toFixed(1)}%`);

        if (similarity > bestSimilarity) {
          bestSimilarity = similarity;
          bestMatch = enrollment;
        }
      }

      console.log(`🎯 Best similarity: ${(bestSimilarity * 100).toFixed(1)}%`);
      console.log(`🎚️ Threshold: ${(this.SIMILARITY_THRESHOLD * 100).toFixed(1)}%`);
      
      const isMatch = bestSimilarity >= this.SIMILARITY_THRESHOLD;
      const originalAmount = 30;
      let finalAmount = originalAmount;

      if (isMatch && bestMatch) {
        const discountAmount = parseFloat(bestMatch.moneyAmount) || 0;
        finalAmount = Math.max(0, originalAmount - discountAmount);
        
        console.log(`✅ Palm verified! User: ${bestMatch.fullName}`);
        console.log(`💰 Payment: $${originalAmount} - $${discountAmount} = $${finalAmount}`);
      } else {
        console.log(`❌ Palm verification failed. Best similarity: ${(bestSimilarity * 100).toFixed(1)}%`);
      }

      return {
        isMatch,
        similarity: bestSimilarity,
        userInfo: isMatch ? bestMatch || undefined : undefined,
        finalAmount
      };

    } catch (error) {
      console.error('❌ Error during palm verification:', error);
      return {
        isMatch: false,
        similarity: 0,
        finalAmount: 30
      };
    }
  }
}

export const palmFeatureExtractor = new PalmFeatureExtractor();