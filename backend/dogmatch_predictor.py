"""
DogMatch Predictor - Classe para integração no backend

Este arquivo contém a classe DogMatchPredictor que pode ser usada
no backend (Flask/FastAPI) para fazer predições de raças de cães.
Sistema híbrido otimizado com feature engineering e busca por similaridade.
"""

import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any


class DogMatchPredictor:
    """
    Classe para predição de raças de cães baseado nas preferências do usuário.
    
    Sistema híbrido otimizado que combina:
    - Modelo principal (KNN_Advanced)
    - Modelo de similaridade (NearestNeighbors)
    - Feature engineering avançado
    - RobustScaler para normalização
    """
    
    def __init__(self):
        """
        Inicializa o preditor carregando todos os modelos e preprocessadores.
        """
        try:
            # Obter caminho absoluto do diretório atual
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(current_dir, 'models')
            
            # Carregar modelos híbridos otimizados (dataset filtrado)
            self.model = joblib.load(os.path.join(models_dir, 'dogmatch_optimized_model.pkl'))
            self.similarity_model = joblib.load(os.path.join(models_dir, 'dogmatch_similarity_model.pkl'))
            self.robust_scaler = joblib.load(os.path.join(models_dir, 'robust_scaler.pkl'))
            self.label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
            self.feature_info = joblib.load(os.path.join(models_dir, 'feature_info_optimized.pkl'))
            
            # Carregar dados processados para similaridade (dataset filtrado)
            self.X_enhanced = joblib.load(os.path.join(models_dir, 'X_enhanced.pkl'))
            self.y_processed = joblib.load(os.path.join(models_dir, 'y_processed.pkl'))
            
            # Extrair informações das features
            self.feature_columns = self.feature_info['feature_columns']
            self.categorical_columns = self.feature_info['categorical_columns']
            self.numeric_columns = self.feature_info['numeric_columns']
            self.breed_names = self.feature_info['breed_names']
            self.group_labels = self.feature_info.get('group_labels', [])
            self.breed_labels = self.feature_info.get('breed_labels', [])
            self.label_group_name = self.feature_info.get('label_group_name', 'Label_Grouped')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Arquivo não encontrado: {e}. Certifique-se de que todos os arquivos .pkl estão no diretório correto.")
        except Exception as e:
            raise Exception(f"Erro ao inicializar o preditor: {e}")
    
    def predict(self, user_input: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """
        Prediz raças de cães baseado nas preferências do usuário.
        Sistema híbrido que combina predição principal + similaridade.
        
        Args:
            user_input: Dicionário com as preferências do usuário
            top_k: Número de raças similares a retornar (padrão: 5)
        
        Returns:
            Dicionário com predições, raças similares e perfil do usuário
            
        Example:
            user_preferences = {
                'Size': 'Medium',
                'Exercise Requirements (hrs/day)': 2.0,
                'Good with Children': 'Yes',
                'Intelligence Rating (1-10)': 7,
                'Training Difficulty (1-10)': 3,
                'Shedding Level': 'Moderate',
                'Health Issues Risk': 'Low',
                'Type': 'Herding',
                'Friendly Rating (1-10)': 8,
                'Life Span': 12,
                'Average Weight (kg)': 20
            }
            
            predictor = DogMatchPredictor()
            results = predictor.predict(user_preferences)
        """
        try:
            k = max(1, min(int(top_k), len(self.breed_names)))
            # Validar entrada
            self._validate_input(user_input)
            
            # Criar DataFrame com a entrada do usuário
            user_df = pd.DataFrame([user_input])
            
            # Aplicar label encoding para variáveis categóricas
            for col in self.categorical_columns:
                if col in user_df.columns:
                    if user_input[col] not in self.label_encoders[col].classes_:
                        raise ValueError(f"Valor inválido para '{col}': {user_input[col]}. "
                                       f"Valores aceitos: {list(self.label_encoders[col].classes_)}")
                    user_df[col] = self.label_encoders[col].transform(user_df[col])
            
            # Criar features derivadas (feature engineering)
            user_df = self._create_derived_features(user_df)
            
            # Reordenar e garantir todas as colunas esperadas pelo modelo
            derived_cols = [
                'Family_Compatibility_Score',
                'Maintenance_Score',
                'Energy_Score',
                'Intelligence_Training_Ratio',
                'Size_Score'
            ]
            expected_columns = list(self.feature_columns)
            for col in derived_cols:
                if col not in expected_columns:
                    expected_columns.append(col)
            user_df = user_df.reindex(columns=expected_columns, fill_value=0)
            
            # Aplicar robust scaling para variáveis numéricas
            user_df[self.numeric_columns] = self.robust_scaler.transform(user_df[self.numeric_columns])
            
            predictions_grouped = self._generate_group_predictions(user_df, k)
            
            # Encontrar raças similares (sugestões de raças) via nearest neighbors
            breed_suggestions = self._find_similar_breeds(user_df, k)
            predictions = breed_suggestions
            similar_breeds = breed_suggestions
            
            # Calcular perfil do usuário
            user_profile = self._calculate_user_profile(user_df)
            
            # Preparar resultados
            results = {
                'predictions': predictions,
                'similar_breeds': similar_breeds,
                'user_profile': user_profile,
                'predictions_grouped': predictions_grouped
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Erro ao fazer predição: {e}")
    
    def _create_derived_features(self, user_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features derivadas usando feature engineering avançado.
        """
        try:
            # Mapeamentos baseados nos encoders para manter consistência
            def _encoded_weights(column: str, name_to_weight: Dict[str, float], default: float = 0.0) -> Dict[int, float]:
                encoder = self.label_encoders.get(column)
                if encoder is None:
                    return {}
                mapping: Dict[int, float] = {}
                for cls in encoder.classes_:
                    weight = name_to_weight.get(cls.strip().lower(), default)
                    encoded = int(encoder.transform([cls])[0])
                    mapping[encoded] = weight
                return mapping

            children_weights = _encoded_weights(
                'Good with Children',
                {
                    'yes': 1.0,
                    'with training': 0.5,
                    'no': 0.0
                },
                default=0.0
            )

            shedding_weights = _encoded_weights(
                'Shedding Level',
                {
                    'low': 0.0,
                    'moderate': 0.5,
                    'high': 1.0,
                    'very high': 1.5
                },
                default=0.5
            )

            health_weights = _encoded_weights(
                'Health Issues Risk',
                {
                    'low': 0.0,
                    'moderate': 0.5,
                    'high': 1.0
                },
                default=0.5
            )

            size_weights = _encoded_weights(
                'Size',
                {
                    'toy': 1.0,
                    'small': 1.0,
                    'small-medium': 1.5,
                    'medium': 2.0,
                    'large': 3.0,
                    'giant': 4.0
                },
                default=2.0
            )

            # Family_Compatibility_Score
            if 'Good with Children' in user_df.columns and 'Friendly Rating (1-10)' in user_df.columns and 'Training Difficulty (1-10)' in user_df.columns:
                children_score = user_df['Good with Children'].map(children_weights)
                user_df['Family_Compatibility_Score'] = (
                    children_score * 0.4 + 
                    user_df['Friendly Rating (1-10)'] * 0.1 + 
                    (10 - user_df['Training Difficulty (1-10)']) * 0.1
                )
            
            # Maintenance_Score
            if 'Shedding Level' in user_df.columns and 'Exercise Requirements (hrs/day)' in user_df.columns and 'Health Issues Risk' in user_df.columns:
                shedding_score = user_df['Shedding Level'].map(shedding_weights)
                health_score = user_df['Health Issues Risk'].map(health_weights)
                user_df['Maintenance_Score'] = (
                    shedding_score * 0.3 + 
                    user_df['Exercise Requirements (hrs/day)'] * 0.2 + 
                    health_score * 0.3
                )
            
            # Energy_Score
            if 'Exercise Requirements (hrs/day)' in user_df.columns and 'Intelligence Rating (1-10)' in user_df.columns:
                user_df['Energy_Score'] = (
                    user_df['Exercise Requirements (hrs/day)'] * 0.4 + 
                    user_df['Intelligence Rating (1-10)'] * 0.1
                )
            
            # Intelligence_Training_Ratio
            if 'Intelligence Rating (1-10)' in user_df.columns and 'Training Difficulty (1-10)' in user_df.columns:
                user_df['Intelligence_Training_Ratio'] = (
                    user_df['Intelligence Rating (1-10)'] / (user_df['Training Difficulty (1-10)'] + 1)
                )
            
            # Size_Score
            if 'Size' in user_df.columns:
                user_df['Size_Score'] = user_df['Size'].map(size_weights)
            
            return user_df
            
        except Exception as e:
            print(f"Aviso: erro ao criar features derivadas: {e}")
            return user_df
    
    def _find_similar_breeds(self, user_df: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Encontra raças similares usando o modelo de similaridade.
        """
        try:
            # Encontrar raças mais similares
            neighbor_count = min(top_k, len(self.X_enhanced))
            distances, indices = self.similarity_model.kneighbors(user_df, n_neighbors=neighbor_count)
            similarities = 1 / (1 + distances[0])
            max_sim = similarities.max() if len(similarities) > 0 else 1
            
            similar_breeds = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if i < top_k:
                    breed_name = self.breed_labels[idx] if self.breed_labels else str(self.y_processed.iloc[idx])
                    group_label = self.group_labels[idx] if self.group_labels else None
                    similarity_raw = 1 / (1 + dist)
                    similarity = similarity_raw / max_sim if max_sim > 0 else 0.0
                    similar_breeds.append({
                        'breed': breed_name,
                        'group': group_label,
                        'similarity': round(float(similarity), 3),
                        'rank': i + 1
                    })
            
            return similar_breeds
            
        except Exception as e:
            print(f"Aviso: erro ao encontrar raças similares: {e}")
            return []
    
    def _calculate_user_profile(self, user_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula o perfil do usuário baseado nas características.
        """
        try:
            profile = {}
            
            # Family Friendly Score
            if 'Family_Compatibility_Score' in user_df.columns:
                profile['family_friendly'] = round(user_df['Family_Compatibility_Score'].iloc[0], 2)
            
            # Energy Level
            if 'Energy_Score' in user_df.columns:
                profile['energy_level'] = round(user_df['Energy_Score'].iloc[0], 2)
            
            # Maintenance Level
            if 'Maintenance_Score' in user_df.columns:
                profile['maintenance_level'] = round(user_df['Maintenance_Score'].iloc[0], 2)
            
            # Intelligence Level
            if 'Intelligence Rating (1-10)' in user_df.columns:
                profile['intelligence_level'] = round(user_df['Intelligence Rating (1-10)'].iloc[0], 1)
            
            # Size Preference
            if 'Size_Score' in user_df.columns:
                profile['size_preference'] = round(user_df['Size_Score'].iloc[0], 1)
            
            return profile
            
        except Exception as e:
            print(f"Aviso: erro ao calcular perfil do usuário: {e}")
            return {}
    
    def _validate_input(self, user_input: Dict[str, Any]) -> None:
        """
        Valida a entrada do usuário.
        
        Args:
            user_input: Dicionário com as preferências do usuário
            
        Raises:
            ValueError: Se a entrada for inválida
        """
        # Verificar se todas as features necessárias estão presentes
        missing_features = set(self.feature_columns) - set(user_input.keys())
        if missing_features:
            raise ValueError(f"Features ausentes: {missing_features}")
        
        # Verificar tipos de dados para variáveis numéricas
        for col in self.numeric_columns:
            if col in user_input:
                try:
                    float(user_input[col])
                except (ValueError, TypeError):
                    raise ValueError(f"'{col}' deve ser um número. Recebido: {user_input[col]}")
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre as features do modelo.
        
        Returns:
            Dicionário com informações das features
        """
        return {
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numeric_columns': self.numeric_columns,
            'categorical_values': {
                col: list(encoder.classes_) 
                for col, encoder in self.label_encoders.items()
            },
            'breed_names': self.breed_names
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo.
        
        Returns:
            Dicionário com informações do modelo
        """
        return {
            'model_type': type(self.model).__name__,
            'similarity_model_type': type(self.similarity_model).__name__,
            'n_features': len(self.feature_columns),
            'n_breeds': len(self.breed_names),
            'label_group_name': self.label_group_name,
            'supports_probabilities': hasattr(self.model, 'predict_proba'),
            'feature_engineering': True,
            'hybrid_system': True
        }

    def _generate_predictions(self, user_df: pd.DataFrame, top_k: int) -> List[Dict[str, Any]]:
        """
        Gera lista ordenada de predições com score normalizado 0-1.
        """
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(user_df)[0]
            top_indices = np.argsort(probas)[::-1][:top_k]
            predictions = []
            for rank, idx in enumerate(top_indices, start=1):
                breed_name_raw = self.model.classes_[idx] if hasattr(self.model, "classes_") else self.breed_names[idx]
                breed_name = str(breed_name_raw).strip() if breed_name_raw is not None else ''
                if not breed_name:
                    continue
                score = float(probas[idx])
                predictions.append({
                    'breed': breed_name,
                    'score': round(score, 4),
                    'rank': rank
                })
            return predictions
        
        # Fallback: usar modelo de similaridade para ordenar por proximidade
        neighbor_count = min(top_k, len(self.X_enhanced))
        distances, indices = self.similarity_model.kneighbors(user_df, n_neighbors=neighbor_count)
        similarities = 1 / (1 + distances[0])
        max_sim = similarities.max() if len(similarities) > 0 else 1
        predictions = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if rank > top_k:
                break
            breed_name_raw = self.y_processed.iloc[idx]
            breed_name = str(breed_name_raw).strip() if breed_name_raw is not None else ''
            if not breed_name:
                continue
            similarity_raw = 1 / (1 + dist)
            score = similarity_raw / max_sim if max_sim > 0 else 0.0
            predictions.append({
                'breed': breed_name,
                'score': round(float(score), 4),
                'rank': rank
            })
        return predictions

    def _generate_group_predictions(self, user_df: pd.DataFrame, top_k: int) -> List[Dict[str, Any]]:
        """
        Gera lista de predições no nível de grupo (Label_Grouped) com score 0-1.
        """
        if not hasattr(self.model, "predict_proba"):
            return []
        probas = self.model.predict_proba(user_df)[0]
        top_indices = np.argsort(probas)[::-1][:top_k]
        preds = []
        for rank, idx in enumerate(top_indices, start=1):
            group_label = self.model.classes_[idx]
            score = float(probas[idx])
            preds.append({
                'group': group_label,
                'score': round(score, 4),
                'rank': rank
            })
        return preds


if __name__ == "__main__":
    example_input = {
        'Size': 'Medium',
        'Exercise Requirements (hrs/day)': 2.0,
        'Good with Children': 'Yes',
        'Intelligence Rating (1-10)': 7,
        'Training Difficulty (1-10)': 3,
        'Shedding Level': 'Moderate',
        'Health Issues Risk': 'Low',
        'Type': 'Herding',
        'Friendly Rating (1-10)': 8,
        'Life Span': 12,
        'Average Weight (kg)': 20
    }
    predictor = DogMatchPredictor()
    results = predictor.predict(example_input)
    print("Predições de exemplo:", results)