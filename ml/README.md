# DogMatch - Sistema Híbrido de Recomendação de Raças de Cães

##  Descrição

O DogMatch é um sistema híbrido de Machine Learning que recomenda raças de cães ideais baseado nas preferências e características desejadas pelo usuário. O sistema utiliza **algoritmos de similaridade** e **feature engineering avançado** para analisar múltiplas características das raças e sugerir as melhores opções.

##  Características

- **Análise de 158 raças de cães** de 40 países diferentes
- **11 características originais** + **5 features derivadas** (16 total)
- **Sistema híbrido otimizado** (KNN_Advanced + NearestNeighbors)
- **Feature engineering avançado** com RobustScaler
- **Métricas de recomendação** (Top-K Accuracy)
- **API REST** pronta para integração com frontend

##  Dataset

O dataset contém informações sobre:
- **Nome e origem** da raça (40 países)
- **Características físicas** (porte, peso, expectativa de vida)
- **Comportamento** (amigabilidade, inteligência, dificuldade de treino)
- **Cuidados necessários** (exercício, grooming, problemas de saúde)

##  Instalação

### 1. Clonar o repositório
```bash
git clone <url-do-repositorio>
cd dogmatch-project/ml
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Executar o notebook de treinamento
```bash
jupyter notebook DogMatch_ML_Pipeline.ipynb
```

### 4. Testar o sistema
```bash
python dogmatch_predictor.py
```

##  Estrutura do Projeto

```
ml/
├── models/                      # Modelos treinados
│   ├── dogmatch_optimized_model.pkl      # Modelo principal (KNN_Advanced)
│   ├── dogmatch_similarity_model.pkl     # Modelo de similaridade
│   ├── robust_scaler.pkl                 # Normalizador robusto
│   ├── label_encoders.pkl                # Encoders categóricos
│   ├── feature_info_optimized.pkl        # Metadados das features
│   ├── X_enhanced.pkl                    # Dados processados
│   └── y_processed.pkl                   # Labels processados
├── data/                        # Dataset
│   └── Dog Breads Around The World.csv   # Dataset original
├── DogMatch_ML_Pipeline.ipynb   # Notebook principal com pipeline de ML
├── dogmatch_predictor.py        # Classe para predições
├── requirements.txt             # Dependências Python
└── README.md                    # Este arquivo
```

##  Como Usar

### 1. Treinamento do Modelo

Execute o notebook `DogMatch_ML_Pipeline.ipynb` para:
- Carregar e explorar o dataset
- Pré-processar os dados com RobustScaler
- Criar features derivadas (feature engineering)
- Treinar sistema híbrido otimizado
- Avaliar com métricas de recomendação (Top-K Accuracy)
- Exportar modelos e preprocessadores

### 2. Fazer Predições

```python
from dogmatch_predictor import DogMatchPredictor

# Inicializar preditor
predictor = DogMatchPredictor()

# Definir preferências do usuário
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

# Fazer predição
results = predictor.predict(user_preferences)

# Resultado
print("Predição principal:", results['predictions'])
print("Raças similares:", results['similar_breeds'])
print("Perfil do usuário:", results['user_profile'])
```

### 3. Integração com Backend

```python
# Para usar em Flask/FastAPI
from dogmatch_predictor import DogMatchPredictor

app = Flask(__name__)
predictor = DogMatchPredictor()

@app.route('/api/recommend', methods=['POST'])
def recommend_breeds():
    user_input = request.json
    results = predictor.predict(user_input)
    return jsonify(results)
```

##  Características Analisadas

| Característica | Tipo | Descrição |
|----------------|------|-----------|
| Size | Categórica | Porte do cão (Small, Medium, Large, Giant) |
| Exercise Requirements | Numérica | Horas de exercício por dia |
| Good with Children | Categórica | Se é bom com crianças (Yes, No, With Training) |
| Intelligence Rating | Numérica | Avaliação de inteligência (1-10) |
| Training Difficulty | Numérica | Dificuldade de treino (1-10) |
| Shedding Level | Categórica | Nível de queda de pelo |
| Health Issues Risk | Categórica | Risco de problemas de saúde |
| Type | Categórica | Tipo de raça (Herding, Working, etc.) |
| Friendly Rating | Numérica | Avaliação de amigabilidade (1-10) |
| Life Span | Numérica | Expectativa de vida em anos |
| Average Weight | Numérica | Peso médio em kg |

## Features Derivadas (Feature Engineering)

| Feature | Descrição | Fórmula |
|---------|-----------|---------|
| Family_Compatibility_Score | Compatibilidade familiar | children_score × 0.4 + friendly × 0.1 + (10-training) × 0.1 |
| Maintenance_Score | Nível de manutenção | shedding × 0.3 + exercise × 0.2 + health × 0.3 |
| Energy_Score | Nível de energia | exercise × 0.4 + intelligence × 0.1 |
| Intelligence_Training_Ratio | Razão inteligência/treino | intelligence / (training + 1) |
| Size_Score | Score de porte | Small=1, Medium=2, Large=3, Giant=4 |

##  Sistema Híbrido Implementado

### **Modelo Principal:**
- **KNN_Advanced** com métricas otimizadas
- **K=3** vizinhos mais próximos
- **Métrica:** Cosine similarity
- **Pesos:** Distance-based

### **Modelo de Similaridade:**
- **NearestNeighbors** para busca por proximidade
- **Top-K** recomendações (padrão: 5)
- **Similaridade** baseada em características

### **Preprocessamento:**
- **RobustScaler** para normalização robusta
- **LabelEncoder** para variáveis categóricas
- **Feature Engineering** com 5 features derivadas

##  Métricas e Avaliação

- Dataset atual: `ml/data/Dog Breads Around The World.csv` (~159 raças, 1 amostra por raça).
- Alvo atual: `Type` (agrupamento de classe). Porte continua como feature.
- Validação k-fold estratificada por `Type` (min_count=2, n_splits=2):
  - Fold1: Top-1=0.8125, Top-3=0.9750, Top-5=0.9875, Top-10=1.0000
  - Fold2: Top-1=0.7722, Top-3=0.9494, Top-5=0.9620, Top-10=1.0000
  - Médias: Top-1≈0.7923, Top-3≈0.9622, Top-5≈0.9748, Top-10≈1.0000
- Similaridade 0–1 não é probabilidade calibrada; é usada para ranquear raças dentro/entre grupos.

##  Racional do Agrupamento

- Cada raça tem 1 amostra; classificar por raça não generaliza (top-k ~0).
- Agrupar por `Type` aumenta exemplos por classe e permite validação estratificada com métricas úteis.
- Porte permanece como feature e é usado na similaridade/ranking para sugerir raças dentro do Type.

##  Limitações

- min_count por `Type` ainda é 2; mais dados por Type tornariam o k-fold mais robusto.
- Por raça, segue 1 amostra: classificação por raça pura não generaliza; recomendações de raça são via similaridade.
- Metadados (descrições, imagens, temperament/care) dependem de curadoria; imagens locais cobrem apenas parte das raças.
- Warnings de scikit-learn podem ocorrer por diferença de versão do pickle; alinhar versão ou regenerar modelos no runtime alvo.

##  Próximos Passos

1. **Dados**: coletar mais exemplos por Type/raça; fundir Types raros se necessário.
2. **Modelo**: opcional testar embeddings + ANN para similaridade; calibrar scores ou usar score de Type na UI para evitar percepção de 100%.
3. **Metadados**: curar/expandir descrições, imagens e temperament/care.
4. **Validação**: revalidar assim que houver mais dados; ajustar buckets de porte se o dataset crescer.

##  Deploy

### **Vercel (Recomendado - 100% Gratuito):**
```bash
# Instalar Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

### **Estrutura para Deploy:**
```
backend/
├── app.py                    # Flask API
├── dogmatch_predictor.py     # Classe ML
├── models/                   # Arquivos .pkl
├── requirements.txt
└── vercel.json
```


##  Contato


Para dúvidas ou sugestões, entre em contato através das issues do repositório.
