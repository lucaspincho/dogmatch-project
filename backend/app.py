"""
DogMatch Backend API - Flask
Sistema híbrido de recomendação de raças de cães
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys
import pandas as pd

# Adicionar o diretório atual ao path para importar o predictor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dogmatch_predictor import DogMatchPredictor

# Inicializar Flask app
app = Flask(__name__)
CORS(app)  # Permitir CORS para frontend

# Carregar modelo uma vez (cache global)
predictor = None
breed_metadata_cache = None

def get_predictor():
    """Carregar predictor com cache"""
    global predictor
    if predictor is None:
        try:
            predictor = DogMatchPredictor()
        except Exception as e:
            print(f"Erro ao carregar predictor: {e}")
            raise e
    return predictor


def load_breed_metadata():
    """
    Carrega metadados das raças a partir do CSV usado no treinamento.
    """
    global breed_metadata_cache
    if breed_metadata_cache is not None:
        return breed_metadata_cache

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "..", "ml", "data", "Dog Breads Around The World.csv")
        df = pd.read_csv(csv_path)

        image_map = {
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
        }

        records = []
        for row in df.to_dict(orient="records"):
            name = row.get("Name")
            good_children_raw = str(row.get("Good with Children")).lower()
            if good_children_raw == "yes":
                good_children_val = True
            elif good_children_raw == "no":
                good_children_val = False
            else:
                good_children_val = None
            records.append({
                "name": name,
                "size": row.get("Size"),
                "breed_group": row.get("Type"),
                "shedding": row.get("Shedding Level"),
                "exercise_needs": row.get("Exercise Requirements (hrs/day)"),
                "good_with_children": good_children_val,
                "intelligence": row.get("Intelligence Rating (1-10)"),
                "training_difficulty": row.get("Training Difficulty (1-10)"),
                "health_risk": row.get("Health Issues Risk"),
                "friendliness": row.get("Friendly Rating (1-10)"),
                "life_expectancy": row.get("Life Span"),
                "average_weight": row.get("Average Weight (kg)"),
                "description": row.get("Unique Feature"),
                "temperament": [],
                "care": [],
                "history": row.get("Origin"),
                "images": [image_map[name]] if name in image_map else []
            })

        breed_metadata_cache = records
        return breed_metadata_cache
    except Exception as exc:
        print(f"Erro ao carregar metadados das raças: {exc}")
        breed_metadata_cache = []
        return breed_metadata_cache

@app.route('/')
def home():
    """Página inicial da API"""
    return jsonify({
        "message": "DogMatch API - Sistema Híbrido de Recomendação",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "POST /api/recommend": "Recomendar raças de cães",
            "GET /api/breeds": "Listar todas as raças",
            "GET /api/health": "Status da API",
            "GET /api/features": "Informações das features",
            "GET /api/model-info": "Informações do modelo"
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Verificar saúde da API"""
    try:
        predictor = get_predictor()
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "message": "API funcionando corretamente"
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e)
        }), 500

@app.route('/api/recommend', methods=['POST'])
def recommend_breeds():
    """Endpoint principal para recomendar raças"""
    try:
        # Validar entrada
        if not request.json:
            return jsonify({"error": "JSON body é obrigatório"}), 400
        
        user_input = request.json
        if not isinstance(user_input, dict):
            return jsonify({"error": "Formato inválido: esperado objeto JSON"}), 400
        
        predictor = get_predictor()
        feature_info = predictor.get_feature_info()
        
        # top_k opcional
        top_k_param = request.args.get("top_k", default=5)
        try:
            top_k = int(top_k_param)
            if top_k <= 0:
                raise ValueError()
        except ValueError:
            return jsonify({"error": "top_k deve ser inteiro positivo"}), 400
        
        # Validar campos obrigatórios
        required_fields = feature_info['feature_columns']
        
        missing_fields = [field for field in required_fields if field not in user_input]
        if missing_fields:
            return jsonify({
                "error": f"Campos obrigatórios ausentes: {missing_fields}",
                "required_fields": required_fields
            }), 400
        
        # Validar tipos numéricos
        numeric_columns = feature_info['numeric_columns']
        for col in numeric_columns:
            if col in user_input:
                try:
                    float(user_input[col])
                except (ValueError, TypeError):
                    return jsonify({"error": f"Campo '{col}' deve ser numérico"}), 400
        
        # Validar valores categóricos
        categorical_values = feature_info.get('categorical_values', {})
        for col, allowed in categorical_values.items():
            if col in user_input and user_input[col] not in allowed:
                return jsonify({
                    "error": f"Valor inválido para '{col}'",
                    "allowed_values": {col: allowed}
                }), 400
        
        # Fazer predição
        results = predictor.predict(user_input, top_k=top_k)
        
        # Adicionar metadados
        results['api_version'] = '1.0.0'
        results['timestamp'] = str(pd.Timestamp.now())
        
        return jsonify(results)
        
    except ValueError as e:
        return jsonify({"error": f"Erro de validação: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

@app.route('/api/breeds', methods=['GET'])
def get_breeds():
    """Listar todas as raças disponíveis"""
    try:
        metadata = load_breed_metadata()
        return jsonify({
            "breeds": metadata,
            "total_breeds": len(metadata),
            "api_version": "1.0.0"
        })
        
    except Exception as e:
        return jsonify({"error": f"Erro ao listar raças: {str(e)}"}), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """Informações sobre as features do modelo"""
    try:
        predictor = get_predictor()
        feature_info = predictor.get_feature_info()
        
        return jsonify({
            "features": {
                "categorical": feature_info['categorical_columns'],
                "numeric": feature_info['numeric_columns'],
                "total": len(feature_info['feature_columns'])
            },
            "categorical_values": feature_info['categorical_values'],
            "api_version": "1.0.0"
        })
        
    except Exception as e:
        return jsonify({"error": f"Erro ao obter features: {str(e)}"}), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Informações sobre o modelo"""
    try:
        predictor = get_predictor()
        model_info = predictor.get_model_info()
        
        return jsonify({
            "model": model_info,
            "api_version": "1.0.0"
        })
        
    except Exception as e:
        return jsonify({"error": f"Erro ao obter informações do modelo: {str(e)}"}), 500

@app.route('/api/example', methods=['GET'])
def get_example():
    """Exemplo de entrada para a API"""
    example_input = {
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
    
    return jsonify({
        "example_input": example_input,
        "description": "Exemplo de entrada para o endpoint /api/recommend",
        "usage": "POST /api/recommend com este JSON no body"
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint não encontrado"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Método não permitido"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Erro interno do servidor"}), 500

if __name__ == '__main__':
    # Configurações para desenvolvimento e produção
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
