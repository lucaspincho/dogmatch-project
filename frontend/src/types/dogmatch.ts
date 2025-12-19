export type DogSize = "Small" | "Medium" | "Large" | "Giant" | "Unknown";
export type SheddingLevel = "Low" | "Moderate" | "High" | "Unknown";
export type HealthRisk = "Low" | "Medium" | "High" | "Moderate" | "Unknown";
export type BreedGroup = "Herding" | "Sporting" | "Working" | "Hound" | "Terrier" | "Toy" | "Non-Sporting" | "Unknown";

export interface UserPreferences {
  size: DogSize;
  exerciseHours: number;
  goodWithChildren: boolean;
  intelligence: number;
  trainingDifficulty: number;
  shedding: SheddingLevel;
  healthRisk: HealthRisk;
  breedGroup: BreedGroup;
  friendliness: number;
  lifeExpectancy: number;
  averageWeight: number;
}

export interface DogBreed {
  id: string;
  name: string;
  size: DogSize;
  exerciseNeeds: number | null;
  goodWithChildren: boolean | null;
  intelligence: number | null;
  trainingDifficulty: number | null;
  shedding: SheddingLevel;
  healthRisk: HealthRisk;
  breedGroup: BreedGroup;
  friendliness: number | null;
  lifeExpectancy: number | null;
  averageWeight: number | null;
  description: string | null;
  temperament: string[];
  care: string[];
  history: string | null;
  images: string[];
}

export interface RecommendationResult {
  breed: DogBreed;
  compatibilityScore: number;
  matchReasons: string[];
  topPredictions: {
    name: string;
    score: number;
  }[];
  similarBreeds: {
    breed: DogBreed;
    similarityScore: number;
  }[];
}

// Tipos para integração com API
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
    similarity: number;
    group?: string;
  }>;
  user_profile: Record<string, number | string | null | undefined>;
  predictions_grouped?: Array<{
    group: string;
    score: number;
    rank: number;
  }>;
}