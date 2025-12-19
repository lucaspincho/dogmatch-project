/**
 * Traduções de campos e valores para português
 */

// Tradução de tamanhos
export const sizeTranslations: Record<string, string> = {
  'Small': 'Pequeno',
  'Small-Medium': 'Pequeno-Médio',
  'Medium': 'Médio',
  'Large': 'Grande',
  'Giant': 'Gigante',
  'Toy': 'Toy',
  'Unknown': 'Desconhecido'
};

// Tradução de níveis de queda de pelo
export const sheddingTranslations: Record<string, string> = {
  'Low': 'Baixa',
  'Moderate': 'Moderada',
  'High': 'Alta',
  'Unknown': 'Desconhecido'
};

// Tradução de riscos de saúde
export const healthRiskTranslations: Record<string, string> = {
  'Low': 'Baixo',
  'Medium': 'Médio',
  'Moderate': 'Moderado',
  'High': 'Alto',
  'Unknown': 'Desconhecido'
};

// Tradução de grupos de raças
export const breedGroupTranslations: Record<string, string> = {
  'Herding': 'Pastoreio',
  'Sporting': 'Esportivo',
  'Working': 'Trabalho',
  'Hound': 'Sabujo',
  'Terrier': 'Terrier',
  'Toy': 'Toy',
  'Non-Sporting': 'Não Esportivo',
  'Unknown': 'Desconhecido'
};

// Tradução de compatibilidade com crianças
export const childrenCompatibilityTranslations: Record<string, string> = {
  'Yes': 'Sim',
  'No': 'Não',
  'With Training': 'Com Treinamento'
};

/**
 * Função para traduzir tamanho
 */
export function translateSize(size: string): string {
  return sizeTranslations[size] || size;
}

/**
 * Função para traduzir queda de pelo
 */
export function translateShedding(shedding: string): string {
  return sheddingTranslations[shedding] || shedding;
}

/**
 * Função para traduzir risco de saúde
 */
export function translateHealthRisk(healthRisk: string): string {
  return healthRiskTranslations[healthRisk] || healthRisk;
}

/**
 * Função para traduzir grupo de raça
 */
export function translateBreedGroup(breedGroup: string): string {
  return breedGroupTranslations[breedGroup] || breedGroup;
}

/**
 * Função para traduzir compatibilidade com crianças
 */
export function translateChildrenCompatibility(compatibility: string | boolean): string {
  if (typeof compatibility === 'boolean') {
    return compatibility ? 'Sim' : 'Não';
  }
  return childrenCompatibilityTranslations[compatibility] || compatibility;
}
