#  DogMatch Backend API

##  Descrição

API Flask para o sistema híbrido de recomendação de raças de cães DogMatch. Fornece endpoints REST para integração com frontend e outras aplicações.

##  Características

- **API REST** completa com Flask
- **Sistema híbrido** de ML integrado
- **CORS** habilitado para frontend
- **Validação** robusta de entrada
- **Error handling** completo
- **Deploy** no Vercel (gratuito)

## Limitações e Métricas

- Dataset atual: `ml/data/Dog Breads Around The World.csv` (~159 raças, 1 amostra/raça). Alvo atual = `Type` (agrupado).
- Validação k-fold estratificada por Type (min_count=2, n_splits=2):
  - Fold1: Top-1=0.8125, Top-3=0.9750, Top-5=0.9875, Top-10=1.0000
  - Fold2: Top-1=0.7722, Top-3=0.9494, Top-5=0.9620, Top-10=1.0000
  - Médias: Top-1≈0.7923, Top-3≈0.9622, Top-5≈0.9748, Top-10≈1.0000
- Racional: por raça há 1 amostra; agrupar por Type permite validação útil. Porte segue como feature e é usado na similaridade/ranking de raças.
- Scores são de similaridade (0–1), não probabilidades calibradas.
- Limitações: min_count por Type ainda é 2; por raça continua 1 (classificação de raça pura não generaliza). Metadados dependem de curadoria e podem faltar imagens. Warnings de versão do scikit-learn podem surgir se pickle ≠ runtime.

##  Estrutura

```
backend/
├── app.py                    # API Flask principal
├── dogmatch_predictor.py     # Classe ML (cópia do ml/)
├── models/                   # Arquivos .pkl
│   ├── dogmatch_optimized_model.pkl
│   ├── dogmatch_similarity_model.pkl
│   ├── robust_scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_info_optimized.pkl
│   ├── X_enhanced.pkl
│   └── y_processed.pkl
├── requirements.txt          # Dependências Python
├── vercel.json              # Configuração Vercel
└── README.md                # Este arquivo
```

## Instalação

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Executar localmente
```bash
python app.py
```

### 3. Testar API
```bash
# Health check
curl http://localhost:5000/api/health

# Exemplo de recomendação
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "Size": "Medium",
    "Exercise Requirements (hrs/day)": 2.0,
    "Good with Children": "Yes",
    "Intelligence Rating (1-10)": 7,
    "Training Difficulty (1-10)": 3,
    "Shedding Level": "Moderate",
    "Health Issues Risk": "Low",
    "Type": "Herding",
    "Friendly Rating (1-10)": 8,
    "Life Span": 12,
    "Average Weight (kg)": 20
  }'
```

## Endpoints

### `GET /`
Página inicial da API com informações básicas.

### `POST /api/recommend`
**Endpoint principal** para recomendar raças.

**Body (JSON):**
```json
{
  "Size": "Medium",
  "Exercise Requirements (hrs/day)": 2.0,
  "Good with Children": "Yes",
  "Intelligence Rating (1-10)": 7,
  "Training Difficulty (1-10)": 3,
  "Shedding Level": "Moderate",
  "Health Issues Risk": "Low",
  "Type": "Herding",
  "Friendly Rating (1-10)": 8,
  "Life Span": 12,
  "Average Weight (kg)": 20
}
```

**Response (valores ilustrativos):**
```json
{
  "predictions": [
    {"breed": "Australian Shepherd", "score": 0.18, "rank": 1},
    {"breed": "Briard", "score": 0.16, "rank": 2},
    {"breed": "Australian Cattle Dog", "score": 0.14, "rank": 3}
  ],
  "similar_breeds": [
    {"breed": "Briard", "similarity": 0.62, "rank": 1},
    {"breed": "Australian Cattle Dog", "similarity": 0.55, "rank": 2}
  ],
  "user_profile": {
    "family_friendly": 0.4,
    "energy_level": 1.2,
    "maintenance_level": 0.3,
    "size_preference": 2.0
  },
  "api_version": "1.0.0",
  "timestamp": "2025-01-03T12:00:00"
}
```

### `GET /api/breeds`
Lista todas as raças disponíveis.

### `GET /api/features`
Informações sobre as features do modelo.

### `GET /api/model-info`
Informações técnicas do modelo.

### `GET /api/health`
Status da API e modelo.

### `GET /api/example`
Exemplo de entrada para a API.

##  Deploy

### Vercel (Recomendado - 100% Gratuito)
```bash
# Instalar Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

### Heroku
```bash
# Criar Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
git push heroku main
```

## Desenvolvimento

### Executar em modo debug
```bash
python app.py
```

### Testar endpoints
```bash
# Usar curl ou Postman
curl http://localhost:5000/api/health
```

##  Monitoramento

### Health Check
```bash
curl http://localhost:5000/api/health
```

### Logs
- **Local:** Console do terminal
- **Vercel:** Dashboard do Vercel
- **Heroku:** `heroku logs --tail`

##  Troubleshooting

### Erro: "Model not found"
- Verificar se arquivos `.pkl` estão na pasta `models/`
- Verificar permissões de leitura

### Erro: "Import error"
- Verificar se `dogmatch_predictor.py` está no diretório
- Verificar dependências instaladas

### Erro: "CORS"
- Verificar se `Flask-CORS` está instalado
- Verificar configuração CORS

##  Suporte

- **GitHub:** [Issues](https://github.com/thiagovonsohsten/dogmatch-project/issues)
- **API Docs:** `/api/example` para exemplos
- **Health:** `/api/health` para status

---


