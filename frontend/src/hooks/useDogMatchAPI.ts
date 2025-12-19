/**
 * Hook personalizado para integração com DogMatch API
 */

import { useState, useCallback } from 'react';
import { dogMatchAPI, ApiUserPreferences, ApiRecommendationResponse } from '@/lib/api';
import { UserPreferences, RecommendationResult } from '@/types/dogmatch';
import { toast } from 'sonner';

export const useDogMatchAPI = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getRecommendation = useCallback(async (preferences: UserPreferences, topK: number = 5): Promise<RecommendationResult | null> => {
    setLoading(true);
    setError(null);

    try {
      // Converte preferências para formato da API
      const apiPreferences = dogMatchAPI.convertPreferencesToApi(preferences);
      
      // Faz requisição para API
      const [apiResponse, breeds] = await Promise.all([
        dogMatchAPI.getRecommendation(apiPreferences, topK),
        dogMatchAPI.getBreeds()
      ]);

      // Converte resposta para formato do frontend
      const result = dogMatchAPI.convertApiResponseToFrontend(apiResponse, breeds);
      
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Erro desconhecido';
      setError(errorMessage);
      toast.error(`Erro ao obter recomendação: ${errorMessage}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const healthCheck = useCallback(async (): Promise<boolean> => {
    try {
      const response = await dogMatchAPI.healthCheck();
      return response.status === 'healthy' && response.model_loaded;
    } catch (err) {
      console.error('Health check failed:', err);
      return false;
    }
  }, []);

  const getBreeds = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const breeds = await dogMatchAPI.getBreeds();
      return breeds;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Erro desconhecido';
      setError(errorMessage);
      toast.error(`Erro ao carregar raças: ${errorMessage}`);
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  const getModelInfo = useCallback(async () => {
    try {
      const info = await dogMatchAPI.getModelInfo();
      return info;
    } catch (err) {
      console.error('Failed to get model info:', err);
      return null;
    }
  }, []);

  return {
    loading,
    error,
    getRecommendation,
    healthCheck,
    getBreeds,
    getModelInfo,
    clearError: () => setError(null)
  };
};
