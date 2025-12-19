/**
 * DogMatch API Service
 * Conecta o frontend com o backend Flask
 */

import type { RecommendationResult, UserPreferences, DogSize, SheddingLevel, HealthRisk, BreedGroup } from "@/types/dogmatch";

const env = (import.meta as ImportMeta & { env?: { VITE_API_URL?: string } }).env;
const API_BASE_URL = env?.VITE_API_URL || "http://127.0.0.1:5000";

export interface ApiUserPreferences {
  Size: string;
  "Exercise Requirements (hrs/day)": number;
  "Good with Children": string;
  "Intelligence Rating (1-10)": number;
  "Training Difficulty (1-10)": number;
  "Shedding Level": string;
  "Health Issues Risk": string;
  Type: string;
  "Friendly Rating (1-10)": number;
  "Life Span": number;
  "Average Weight (kg)": number;
}

export interface ApiRecommendationResponse {
  api_version: string;
  predictions: Array<{
    breed: string;
    score: number;
  }>;
  similar_breeds: Array<{
    breed: string;
    rank: number;
    similarity: number;  // Campo correto da API
    group?: string;
  }>;
  user_profile: Record<string, number | string | null | undefined>;
  predictions_grouped?: Array<{
    group: string;
    score: number;
    rank: number;
  }>;
}

export interface ApiFeaturesResponse {
  features: {
    categorical: string[];
    numeric: string[];
    total: number;
  };
  categorical_values: Record<string, string[]>;
  api_version: string;
}

export interface ApiBreed {
  name: string;
  size?: string;
  exercise_needs?: number;
  good_with_children?: boolean;
  intelligence?: number;
  training_difficulty?: number;
  shedding?: string;
  health_risk?: string;
  breed_group?: string;
  friendliness?: number;
  life_expectancy?: number;
  average_weight?: number;
  description?: string;
  temperament?: string[];
  care?: string[];
  history?: string;
  images?: string[];
}

export interface ApiHealthResponse {
  message: string;
  model_loaded: boolean;
  status: string;
}

export interface ApiModelInfoResponse {
  model_type: string;
  features: string[];
  total_breeds: number;
  system_type: string;
  accuracy: string;
}

class DogMatchAPI {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    console.log(`API Request: ${options.method || 'GET'} ${url}`);
    if (options.body) {
      console.log('Request Body:', options.body);
    }
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    };

    const response = await fetch(url, { ...defaultOptions, ...options });

    console.log(`API Response: ${response.status} ${response.statusText}`);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('API Error:', errorText);
      throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const data = await response.json();
    console.log('API Data:', data);
    return data;
  }

  /**
   * Verifica se a API está funcionando
   */
  async healthCheck(): Promise<ApiHealthResponse> {
    return this.request<ApiHealthResponse>('/api/health');
  }

  /**
   * Obtém informações do modelo ML
   */
  async getModelInfo(): Promise<ApiModelInfoResponse> {
    return this.request<ApiModelInfoResponse>('/api/model-info');
  }

  /**
   * Obtém lista de todas as raças
   */
  async getBreeds(): Promise<ApiBreed[]> {
    const response = await this.request<{api_version: string, breeds: ApiBreed[]}>('/api/breeds');
    
    const breeds: ApiBreed[] = response.breeds.map((breedMeta) => {
      const images = breedMeta.images && breedMeta.images.length > 0
        ? breedMeta.images
        : this.getBreedImages(breedMeta.name);
      return {
        ...breedMeta,
        images,
      };
    });

    console.log('Breeds carregadas:', breeds.length);
    return breeds;
  }

  private getBreedImages(breedName: string): string[] {
    // Usar imagens locais da pasta public/dog_breeds_img/
    const breedImageMap: { [key: string]: string } = {
      'Beagle': '/dog_breeds_img/beagle.jpg',
      'Border Collie': '/dog_breeds_img/border collie.jpg',
      'Bull Terrier': '/dog_breeds_img/bull terrier.jpg',
      'Chihuahua': '/dog_breeds_img/chihuahua.jpg',
      'Chow Chow': '/dog_breeds_img/chow chow.jpg',
      'Cocker Spaniel': '/dog_breeds_img/cocker spaniel.jpg',
      'Golden Retriever': '/dog_breeds_img/golden.jpg',
      'Siberian Husky': '/dog_breeds_img/husky.jpg',
      'Labrador Retriever': '/dog_breeds_img/labrador.jpg',
      'Lhasa Apso': '/dog_breeds_img/lhasa apso.jpeg',
      'Maltese': '/dog_breeds_img/maltese.jpg',
      'German Shepherd': '/dog_breeds_img/german Shepherd.jpg',
      'Miniature Pinscher': '/dog_breeds_img/pinscher.jpg',
      'Poodle (Standard)': '/dog_breeds_img/poodle.jpg',
      'Pug': '/dog_breeds_img/pug.jpg',
      'Rottweiler': '/dog_breeds_img/rotweiller.jpg',
      'Samoyed': '/dog_breeds_img/samoyed.jpg',
      'Saint Bernard': '/dog_breeds_img/saint bernard.jpg',
      'Standard Schnauzer': '/dog_breeds_img/schnauzer.jpg',
      'Shih Tzu': '/dog_breeds_img/shih tzu.jpeg',
      'Dachshund': '/dog_breeds_img/dachshund.jpg',
      'West Highland White Terrier': '/dog_breeds_img/west highland white terrier.jpg',
      'Yorkshire Terrier': '/dog_breeds_img/yorkshire.jpg',
      'English Bulldog': '/dog_breeds_img/english bulldog.jpg',
      'French Bulldog': '/dog_breeds_img/french bulldog.jpg',
    };

    // Retornar imagem local se disponível, senão fallback
    if (breedImageMap[breedName]) {
      return [breedImageMap[breedName]];
    }

    // Fallback para beagle se não encontrar a raça
    return ['/dog_breeds_img/beagle.jpg'];
  }

  /**
   * Obtém informações das features
   */
  async getFeatures(): Promise<ApiFeaturesResponse> {
    return this.request<ApiFeaturesResponse>('/api/features');
  }

  /**
   * Envia preferências e recebe recomendação
   */
  async getRecommendation(preferences: ApiUserPreferences, topK: number = 5): Promise<ApiRecommendationResponse> {
    const query = topK ? `?top_k=${topK}` : '';
    return this.request<ApiRecommendationResponse>(`/api/recommend${query}`, {
      method: 'POST',
      body: JSON.stringify(preferences),
    });
  }

  /**
   * Converte preferências do frontend para formato da API
   */
  convertPreferencesToApi(frontendPrefs: UserPreferences): ApiUserPreferences {
    console.log('Convertendo preferências do frontend:', frontendPrefs);
    
    // Mapear valores do frontend para valores aceitos pela API
    const mapHealthRisk = (risk: string) => {
      switch (risk) {
        case 'Low': return 'Low';
        case 'Medium': return 'Moderate';  // Frontend usa 'Medium', API usa 'Moderate'
        case 'High': return 'High';
        default: return 'Moderate';
      }
    };

    const mapSheddingLevel = (level: string) => {
      switch (level) {
        case 'Low': return 'Low';
        case 'Moderate': return 'Moderate';
        case 'High': return 'High';
        default: return 'Moderate';
      }
    };

    const apiPrefs = {
      Size: frontendPrefs.size,
      "Exercise Requirements (hrs/day)": frontendPrefs.exerciseHours,
      "Good with Children": frontendPrefs.goodWithChildren ? "Yes" : "No",
      "Intelligence Rating (1-10)": frontendPrefs.intelligence,
      "Training Difficulty (1-10)": frontendPrefs.trainingDifficulty,
      "Shedding Level": mapSheddingLevel(frontendPrefs.shedding),
      "Health Issues Risk": mapHealthRisk(frontendPrefs.healthRisk),
      Type: frontendPrefs.breedGroup,
      "Friendly Rating (1-10)": frontendPrefs.friendliness,
      "Life Span": frontendPrefs.lifeExpectancy,
      "Average Weight (kg)": frontendPrefs.averageWeight,
    };

    console.log('Preferências convertidas para API:', apiPrefs);
    return apiPrefs;
  }

  /**
   * Converte resposta da API para formato do frontend
   */
  convertApiResponseToFrontend(apiResponse: ApiRecommendationResponse, breeds: ApiBreed[]): RecommendationResult {
    const bestPrediction = apiResponse.predictions?.[0];
    
    if (!bestPrediction) {
      throw new Error('Nenhuma predição encontrada na resposta da API');
    }

    const recommendedBreed = this.findBreedOrFallback(bestPrediction.breed || '', breeds);

    const bestBreedNameStr = this.safeToString(bestPrediction?.breed);
    
    const similarBreeds = (apiResponse.similar_breeds || [])
      .filter(similar => {
        const similarBreedName = this.safeToString(similar?.breed);
        if (!similarBreedName || !bestBreedNameStr) return false;
        return !this.safeBreedNameCompare(similarBreedName, bestBreedNameStr);
      })
      .map(similar => {
        const breedName = this.safeToString(similar?.breed);
        const apiBreed = this.findBreedOrFallback(breedName, breeds);
        const breed = this.convertApiBreedToDogBreed(apiBreed);
        return {
          breed,
          similarityScore: this.clampPercentage((similar?.similarity ?? 0) * 100)
        };
      });

    const similarityForBest = apiResponse.similar_breeds?.find((item) => {
      const itemBreed = this.safeToString(item?.breed);
      return this.safeBreedNameCompare(itemBreed, bestBreedNameStr);
    });

    // Compatibilidade: prioriza score de grupo; senão, usa similaridade reescalada com teto
    const topGroupScore = apiResponse.predictions_grouped?.[0]?.score ?? null;
    const sims = (apiResponse.similar_breeds || []).map((s) => s?.similarity ?? 0).filter((v) => Number.isFinite(v));
    const bestSimilarity = sims.length > 0 ? Math.max(...sims) : similarityForBest?.similarity ?? 0;
    const meanTopK =
      sims.length > 0 ? sims.slice(0, Math.min(5, sims.length)).reduce((a, b) => a + b, 0) / Math.min(5, sims.length) : 0;

    let compatibilityScore: number;
    if (typeof topGroupScore === "number") {
      // Teto para evitar 100%: cap em 95
      compatibilityScore = this.clampPercentage(topGroupScore * 100);
      compatibilityScore = Math.min(compatibilityScore, 95);
    } else {
      const adjusted = meanTopK > 0 ? bestSimilarity / (meanTopK + 1e-6) : bestSimilarity;
      compatibilityScore = this.clampPercentage(adjusted * 100);
      compatibilityScore = Math.min(compatibilityScore, 95);
    }

    const matchReasons = this.generateMatchReasons(apiResponse.user_profile, recommendedBreed);
    const topPredictions = (apiResponse.predictions || [])
      .filter(pred => {
        const breedName = this.safeToString(pred?.breed).trim();
        return breedName.length > 0;
      })
      .map((pred) => {
        const breedName = this.safeToString(pred?.breed).trim();
        return {
          name: breedName || 'Raça Desconhecida',
          score: this.clampPercentage((pred?.score ?? 0) * 100)
        };
      });

    const breed = this.convertApiBreedToDogBreed(recommendedBreed);
    
    return {
      breed,
      compatibilityScore,
      matchReasons,
      similarBreeds,
      topPredictions
    };
  }

  /**
   * Gera razões de compatibilidade baseadas no perfil do usuário
   */
  private generateMatchReasons(
    userProfile: Record<string, number | string | null | undefined>,
    breed: ApiBreed
  ): string[] {
    const reasons: string[] = [];

    const userSizePref = this.normalizeSizePreference(userProfile?.size_preference);
    const breedSize = this.normalizeSizeLabel(breed.size);
    if (userSizePref && breedSize && userSizePref === breedSize) {
      reasons.push(`Porte ${breedSize.toLowerCase()} alinhado à preferência informada`);
    }

    if (typeof userProfile?.family_friendly === 'number' && userProfile.family_friendly >= 0.6) {
      reasons.push("Compatibilidade familiar favorável segundo o perfil informado");
    }

    if (
      typeof userProfile?.energy_level === 'number' &&
      typeof breed.exercise_needs === 'number'
    ) {
      if (userProfile.energy_level >= 1.5 && breed.exercise_needs >= 2) {
        reasons.push("Nível de energia adequado para rotinas ativas");
      } else if (userProfile.energy_level < 1.5 && breed.exercise_needs <= 1.5) {
        reasons.push("Baixa demanda de exercício combinando com rotina informada");
      }
    }

    if (
      typeof userProfile?.maintenance_level === 'number' &&
      breed.shedding &&
      typeof breed.shedding === 'string' &&
      userProfile.maintenance_level <= 1.5 &&
      breed.shedding.toLowerCase() === "low"
    ) {
      reasons.push("Baixa manutenção compatível com preferência informada");
    }

    if (reasons.length === 0) {
      reasons.push("Recomendação baseada nas características fornecidas");
    }

    return reasons.slice(0, 3);
  }

  private findBreedOrFallback(breedName: string, breeds: ApiBreed[]): ApiBreed {
    const normalizedName = this.safeToString(breedName).trim();
    if (!normalizedName) {
      return {
        name: 'Raça Desconhecida',
        size: 'Medium',
        breed_group: 'Unknown',
        images: this.getBreedImages('Beagle')
      };
    }
    const breed = breeds.find(
      (b) => {
        const breedNameStr = this.safeToString(b?.name).trim();
        return breedNameStr && this.safeBreedNameCompare(breedNameStr, normalizedName);
      }
    );
    if (breed) {
      const normalizedSize = this.normalizeSizeLabel(breed.size) || 'Medium';
      return { ...breed, size: normalizedSize };
    }
    return {
      name: breedName,
      size: 'Medium',
      breed_group: 'Unknown',
      images: this.getBreedImages(breedName)
    };
  }

  private normalizeSizeLabel(size?: string): string | null {
    if (!size) return null;
    const normalized = size.trim().toLowerCase();
    if (normalized.includes('small')) return 'Small';
    if (normalized.includes('medium')) return 'Medium';
    if (normalized.includes('large')) return 'Large';
    if (normalized.includes('giant')) return 'Giant';
    return 'Unknown';
  }

  private normalizeSizePreference(value: number | string | null | undefined): string | null {
    if (typeof value === 'number') {
      if (value < 1.5) return 'Small';
      if (value < 2.5) return 'Medium';
      if (value < 3.5) return 'Large';
      return 'Giant';
    }
    if (typeof value === 'string') {
      return this.normalizeSizeLabel(value);
    }
    return null;
  }

  private clampPercentage(value: number): number {
    const safe = Number.isFinite(value) ? value : 0;
    return Math.max(0, Math.min(100, Math.round(safe)));
  }

  /**
   * Normaliza um valor para string, garantindo que seja seguro chamar toLowerCase()
   */
  private safeToString(value: unknown): string {
    if (value === null || value === undefined) return '';
    if (typeof value === 'string') return value;
    return String(value);
  }

  /**
   * Compara dois nomes de raças de forma segura (case-insensitive)
   */
  private safeBreedNameCompare(name1: unknown, name2: unknown): boolean {
    const str1 = this.safeToString(name1).toLowerCase().trim();
    const str2 = this.safeToString(name2).toLowerCase().trim();
    return str1 === str2 && str1.length > 0;
  }

  /**
   * Converte string para DogSize de forma segura
   */
  private toDogSize(value: string | null | undefined): DogSize {
    if (!value) return 'Unknown';
    const normalized = this.safeToString(value).trim();
    const lower = normalized.toLowerCase();
    if (lower.includes('small')) return 'Small';
    if (lower.includes('medium')) return 'Medium';
    if (lower.includes('large')) return 'Large';
    if (lower.includes('giant')) return 'Giant';
    return 'Unknown';
  }

  /**
   * Converte string para SheddingLevel de forma segura
   */
  private toSheddingLevel(value: string | null | undefined): SheddingLevel {
    if (!value) return 'Unknown';
    const normalized = this.safeToString(value).trim();
    const lower = normalized.toLowerCase();
    if (lower.includes('low')) return 'Low';
    if (lower.includes('moderate')) return 'Moderate';
    if (lower.includes('high')) return 'High';
    return 'Unknown';
  }

  /**
   * Converte string para HealthRisk de forma segura
   */
  private toHealthRisk(value: string | null | undefined): HealthRisk {
    if (!value) return 'Unknown';
    const normalized = this.safeToString(value).trim();
    const lower = normalized.toLowerCase();
    if (lower.includes('low')) return 'Low';
    if (lower.includes('moderate')) return 'Moderate';
    if (lower.includes('medium')) return 'Medium';
    if (lower.includes('high')) return 'High';
    return 'Unknown';
  }

  /**
   * Converte string para BreedGroup de forma segura
   */
  private toBreedGroup(value: string | null | undefined): BreedGroup {
    if (!value) return 'Unknown';
    const normalized = this.safeToString(value).trim();
    const lower = normalized.toLowerCase();
    if (lower.includes('herding')) return 'Herding';
    if (lower.includes('sporting')) return 'Sporting';
    if (lower.includes('working')) return 'Working';
    if (lower.includes('hound')) return 'Hound';
    if (lower.includes('terrier')) return 'Terrier';
    if (lower.includes('toy')) return 'Toy';
    if (lower.includes('non-sporting') || lower.includes('nonsporting')) return 'Non-Sporting';
    return 'Unknown';
  }

  /**
   * Converte ApiBreed para DogBreed
   */
  private convertApiBreedToDogBreed(apiBreed: ApiBreed): import("@/types/dogmatch").DogBreed {
    const breedName = this.safeToString(apiBreed.name).trim() || 'Raça Desconhecida';
    return {
      id: breedName.toLowerCase().replace(/\s+/g, '-'),
      name: breedName,
      size: this.toDogSize(apiBreed.size),
      exerciseNeeds: apiBreed.exercise_needs ?? null,
      goodWithChildren: apiBreed.good_with_children ?? null,
      intelligence: apiBreed.intelligence ?? null,
      trainingDifficulty: apiBreed.training_difficulty ?? null,
      shedding: this.toSheddingLevel(apiBreed.shedding),
      healthRisk: this.toHealthRisk(apiBreed.health_risk),
      breedGroup: this.toBreedGroup(apiBreed.breed_group),
      friendliness: apiBreed.friendliness ?? null,
      lifeExpectancy: apiBreed.life_expectancy ?? null,
      averageWeight: apiBreed.average_weight ?? null,
      description: apiBreed.description ?? null,
      temperament: apiBreed.temperament || [],
      care: apiBreed.care || [],
      history: apiBreed.history ?? null,
      images: apiBreed.images || this.getBreedImages(breedName)
    };
  }
}

// Instância singleton da API
export const dogMatchAPI = new DogMatchAPI();

// Hook personalizado para usar a API
export const useDogMatchAPI = () => {
  return dogMatchAPI;
};
