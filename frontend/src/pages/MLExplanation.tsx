import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  ArrowLeft, 
  Database, 
  Settings, 
  Brain, 
  TrendingUp, 
  Target,
  Sparkles,
  BarChart3,
  Zap,
  CheckCircle2
} from "lucide-react";
import Header from "@/components/Header";

export default function MLExplanation() {
  const steps = [
    {
      id: 1,
      title: "Coleta de Dados",
      icon: Database,
      description: "Você respondeu um questionário com suas preferências sobre raças de cães.",
      details: [
        "11 características principais foram coletadas",
        "Tamanho, exercício, compatibilidade com crianças",
        "Inteligência, dificuldade de treino, nível de queda de pelo",
        "Risco de saúde, tipo de raça, amigabilidade",
        "Expectativa de vida e peso médio"
      ],
      color: "text-blue-500"
    },
    {
      id: 2,
      title: "Pré-processamento",
      icon: Settings,
      description: "Seus dados são preparados para o modelo de Machine Learning.",
      details: [
        "Variáveis categóricas são convertidas em números (Label Encoding)",
        "Exemplo: 'Small' → 0, 'Medium' → 1, 'Large' → 2, 'Giant' → 3",
        "Validação de todos os valores de entrada",
        "Estruturação dos dados no formato esperado pelo modelo"
      ],
      color: "text-purple-500"
    },
    {
      id: 3,
      title: "Feature Engineering",
      icon: Sparkles,
      description: "Criamos 5 features derivadas que capturam padrões complexos.",
      details: [
        "Family_Compatibility_Score: Compatibilidade familiar (crianças + amigabilidade + facilidade de treino)",
        "Maintenance_Score: Nível de manutenção (queda de pelo + exercício + saúde)",
        "Energy_Score: Nível de energia (exercício + inteligência)",
        "Intelligence_Training_Ratio: Razão entre inteligência e dificuldade de treino",
        "Size_Score: Score numérico do porte do cão"
      ],
      color: "text-pink-500"
    },
    {
      id: 4,
      title: "Normalização",
      icon: BarChart3,
      description: "Aplicamos RobustScaler para normalizar os dados.",
      details: [
        "Garante que todas as características tenham a mesma escala",
        "RobustScaler é resistente a outliers (valores extremos)",
        "Melhora a performance do algoritmo de Machine Learning",
        "Prepara os dados para comparação precisa entre raças"
      ],
      color: "text-green-500"
    },
    {
      id: 5,
      title: "Modelo Principal (KNN)",
      icon: Brain,
      description: "O algoritmo K-Nearest Neighbors encontra a raça mais próxima das suas preferências.",
      details: [
        "KNN (K=3): Encontra os 3 vizinhos mais próximos no espaço de características",
        "Usa distância cosseno para medir similaridade",
        "Pesos baseados em distância: raças mais próximas têm mais peso",
        "Retorna a raça que melhor corresponde ao seu perfil"
      ],
      color: "text-orange-500"
    },
    {
      id: 6,
      title: "Modelo de Similaridade",
      icon: TrendingUp,
      description: "Buscamos raças similares usando NearestNeighbors.",
      details: [
        "Encontra as 5 raças mais similares à sua preferência",
        "Calcula similaridade baseada em todas as características",
        "Converte distância em score de similaridade (0-100%)",
        "Oferece alternativas caso a raça principal não seja ideal"
      ],
      color: "text-cyan-500"
    },
    {
      id: 7,
      title: "Cálculo do Score",
      icon: Target,
      description: "Calculamos a compatibilidade final como porcentagem.",
      details: [
        "Score baseado na proximidade no espaço de características",
        "Quanto mais próximo, maior a compatibilidade",
        "Considera todas as 16 features (11 originais + 5 derivadas)",
        "Resultado final: porcentagem de compatibilidade (0-100%)"
      ],
      color: "text-red-500"
    }
  ];

  const technicalDetails = {
    dataset: "158 raças de cães de 40 países diferentes",
    features: "11 características originais + 5 features derivadas = 16 features totais",
    models: "Sistema híbrido: KNN_Advanced + NearestNeighbors",
    preprocessing: "RobustScaler + LabelEncoder + Feature Engineering",
    accuracy: "Top-1 ≈ 0% (158 classes) | Top-5 ≈ 10% | Top-10 ≈ 20%"
  };

  return (
    <div className="min-h-screen bg-muted/30">
      <Header />
      
      <div className="pt-24 pb-12 px-4">
        <div className="container mx-auto max-w-5xl">
          {/* Header */}
          <div className="mb-8">
            <Link to="/results">
              <Button variant="ghost" className="mb-6">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Voltar aos Resultados
              </Button>
            </Link>
            
            <div className="text-center mb-8">
              <div className="flex items-center justify-center gap-3 mb-4">
                <Brain className="w-10 h-10 text-primary" />
                <h1 className="text-4xl md:text-5xl font-bold">
                  Como Funciona o Machine Learning
                </h1>
              </div>
              <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                Entenda como suas preferências são processadas e transformadas em recomendações,
                quais métricas usamos (top-k) e por que resultados de baixa confiança podem ocorrer.
              </p>
            </div>
          </div>

          {/* Technical Overview */}
          <Card className="p-6 mb-8 bg-gradient-to-br from-primary/10 to-secondary/10">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <Zap className="w-6 h-6" />
              Visão Geral Técnica
            </h2>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground mb-1">Dataset</p>
                <p className="font-semibold">{technicalDetails.dataset}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground mb-1">Features</p>
                <p className="font-semibold">{technicalDetails.features}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground mb-1">Modelos</p>
                <p className="font-semibold">{technicalDetails.models}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground mb-1">Pré-processamento</p>
                <p className="font-semibold">{technicalDetails.preprocessing}</p>
              </div>
            </div>
          </Card>

          {/* Métricas e limitações */}
          <Card className="p-6 mb-8">
            <h2 className="text-2xl font-bold mb-4">Métricas e limitações</h2>
            <div className="space-y-3 text-sm text-muted-foreground">
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-primary mt-0.5" />
                <span>Top-1 accuracy ≈ 0% para 158 classes; Top-5 ≈ 10%; Top-10 ≈ 20%.</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-primary mt-0.5" />
                <span>Top-k é a métrica principal porque, com muitas classes e poucos exemplos por raça, a chance de acerto exato (top-1) é baixa.</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-primary mt-0.5" />
                <span>Scores representam similaridade relativa, não probabilidade calibrada; use-os como ordenação, não como certeza.</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-primary mt-0.5" />
                <span>Limitações: dataset pequeno e desbalanceado; algumas raças têm poucos exemplos, o que reduz confiança.</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-primary mt-0.5" />
                <span>Implicação: recomendações podem ter baixa confiança; ajuste preferências e avalie as alternativas do top-k.</span>
              </div>
            </div>
          </Card>

          {/* Process Steps */}
          <div className="space-y-6">
            <h2 className="text-3xl font-bold mb-6 text-center">
              Processo Passo a Passo
            </h2>
            
            {steps.map((step, index) => {
              const Icon = step.icon;
              return (
                <Card key={step.id} className="p-6 shadow-medium">
                  <div className="flex items-start gap-4">
                    {/* Step Number */}
                    <div className={`flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center ${
                      step.id === 1 ? 'bg-blue-500/10' :
                      step.id === 2 ? 'bg-purple-500/10' :
                      step.id === 3 ? 'bg-pink-500/10' :
                      step.id === 4 ? 'bg-green-500/10' :
                      step.id === 5 ? 'bg-orange-500/10' :
                      step.id === 6 ? 'bg-cyan-500/10' :
                      'bg-red-500/10'
                    }`}>
                      <Icon className={`w-6 h-6 ${step.color}`} />
                    </div>
                    
                    {/* Content */}
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <Badge variant="outline" className="text-sm">
                          Passo {step.id}
                        </Badge>
                        <h3 className="text-2xl font-bold">{step.title}</h3>
                      </div>
                      
                      <p className="text-muted-foreground mb-4 text-lg">
                        {step.description}
                      </p>
                      
                      <div className="space-y-2">
                        {step.details.map((detail, detailIndex) => (
                          <div key={detailIndex} className="flex items-start gap-2">
                            <CheckCircle2 className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                            <span className="text-sm text-muted-foreground">{detail}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                  
                  {/* Connector Line */}
                  {index < steps.length - 1 && (
                    <div className="flex justify-center mt-4">
                      <div className="w-0.5 h-8 bg-border"></div>
                    </div>
                  )}
                </Card>
              );
            })}
          </div>

          {/* Feature Engineering Details */}
          <Card className="p-6 mt-8 bg-gradient-to-br from-secondary/10 to-primary/10">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <Sparkles className="w-6 h-6" />
              Feature Engineering Detalhado
            </h2>
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2">Family_Compatibility_Score</h4>
                <p className="text-sm text-muted-foreground">
                  Combina compatibilidade com crianças (40%), amigabilidade (10%) e facilidade de treino (10%). 
                  Quanto maior o score, melhor para famílias.
                </p>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Maintenance_Score</h4>
                <p className="text-sm text-muted-foreground">
                  Calcula o nível de manutenção necessário considerando queda de pelo (30%), 
                  exercício diário (20%) e risco de problemas de saúde (30%).
                </p>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Energy_Score</h4>
                <p className="text-sm text-muted-foreground">
                  Mede o nível de energia do cão baseado em horas de exercício (40%) e 
                  inteligência (10%). Cães mais inteligentes geralmente precisam de mais estímulo.
                </p>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Intelligence_Training_Ratio</h4>
                <p className="text-sm text-muted-foreground">
                  Razão entre inteligência e dificuldade de treino. Valores altos indicam 
                  cães inteligentes e fáceis de treinar.
                </p>
              </div>
            </div>
          </Card>

          {/* Why This Approach */}
          <Card className="p-6 mt-8">
            <h2 className="text-2xl font-bold mb-4">Por que essa Abordagem?</h2>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-primary mt-0.5 flex-shrink-0" />
                <div>
                  <p className="font-semibold mb-1">Sistema Híbrido</p>
                  <p className="text-sm text-muted-foreground">
                    Combina predição principal (KNN) com busca por similaridade, oferecendo 
                    não apenas a melhor opção, mas também alternativas relevantes.
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-primary mt-0.5 flex-shrink-0" />
                <div>
                  <p className="font-semibold mb-1">Feature Engineering</p>
                  <p className="text-sm text-muted-foreground">
                    Features derivadas capturam padrões complexos que não são óbvios nas 
                    características originais, melhorando a precisão das recomendações.
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-primary mt-0.5 flex-shrink-0" />
                <div>
                  <p className="font-semibold mb-1">RobustScaler</p>
                  <p className="text-sm text-muted-foreground">
                    Normalização robusta garante que todas as características tenham o mesmo 
                    peso, mesmo na presença de valores extremos.
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-primary mt-0.5 flex-shrink-0" />
                <div>
                  <p className="font-semibold mb-1">Interpretabilidade</p>
                  <p className="text-sm text-muted-foreground">
                    O sistema fornece scores de compatibilidade e razões para a recomendação, 
                    tornando o processo transparente e confiável.
                  </p>
                </div>
              </div>
            </div>
          </Card>

          {/* Back Button */}
          <div className="mt-8 text-center">
            <Link to="/results">
              <Button size="lg" className="bg-gradient-hero">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Voltar aos Resultados
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

