import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ArrowLeft, Heart, Share2, Sparkles, Loader2, Brain, AlertTriangle } from "lucide-react";
import { UserPreferences } from "@/types/dogmatch";
import { RecommendationResult } from "@/types/dogmatch";
import { useDogMatchAPI } from "@/hooks/useDogMatchAPI";
import Header from "@/components/Header";
import BreedImage from "@/components/BreedImage";
import { getBreedNamePT } from "@/lib/breedNames";
import { toast } from "sonner";

export default function Results() {
  const navigate = useNavigate();
  const [result, setResult] = useState<RecommendationResult | null>(null);
  const { loading, error, getRecommendation } = useDogMatchAPI();

  useEffect(() => {
    const fetchRecommendation = async () => {
      const storedPreferences = localStorage.getItem("dogmatch-preferences");
      
      if (!storedPreferences) {
        navigate("/questionnaire");
        return;
      }

      try {
        const preferences: UserPreferences = JSON.parse(storedPreferences);
        const recommendation = await getRecommendation(preferences);
        
        if (recommendation) {
          setResult(recommendation);
        } else {
          toast.error("Não foi possível obter a recomendação. Tente novamente.");
        }
      } catch (err) {
        console.error("Erro ao obter recomendação:", err);
        toast.error("Erro ao processar suas preferências.");
      }
    };

    fetchRecommendation();
  }, [navigate, getRecommendation]);

  const handleShare = async () => {
    try {
      // Criar link para compartilhar os resultados
      const currentUrl = window.location.href;
      const shareData = {
        title: "DogMatch - Resultado de recomendação",
        text: `Recomendação principal: ${result?.breed.name || 'Raça recomendada'}. Veja detalhes e ajuste suas preferências no DogMatch.`,
        url: currentUrl
      };

      // Verificar se a API de compartilhamento está disponível
      if (navigator.share && navigator.canShare && navigator.canShare(shareData)) {
        await navigator.share(shareData);
        toast.success("Compartilhado com sucesso!");
      } else {
        // Fallback: copiar para área de transferência
        await navigator.clipboard.writeText(`${shareData.text}\n\n${shareData.url}`);
        toast.success("Link copiado para a área de transferência!");
      }
    } catch (error) {
      console.error("Erro ao compartilhar:", error);
      toast.error("Erro ao compartilhar. Tente novamente.");
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-muted/30 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-lg">Analisando suas preferências...</p>
          <p className="text-sm text-muted-foreground">Gerando recomendação baseada nos dados fornecidos</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-muted/30 flex items-center justify-center">
        <div className="text-center">
          <p className="text-lg text-destructive mb-4">Erro ao carregar recomendação</p>
          <p className="text-sm text-muted-foreground mb-6">{error}</p>
          <Button onClick={() => navigate("/questionnaire")}>
            Tentar Novamente
          </Button>
        </div>
      </div>
    );
  }

  if (!result) {
    return null;
  }

  return (
    <div className="min-h-screen bg-muted/30">
      <Header />
      
      <div className="pt-24 pb-12 px-4">
        <div className="container mx-auto max-w-6xl">
          <Button
            variant="ghost"
            onClick={() => navigate("/questionnaire")}
            className="mb-6"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Ajustar Preferências
          </Button>

          {/* Main Result Card */}
          <Card className="p-8 md:p-12 mb-8 shadow-strong animate-scale-in">
            <div className="flex items-center gap-3 mb-6">
              <div className="px-4 py-2 bg-gradient-accent rounded-full flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                <span className="font-semibold">Recomendação principal</span>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-8 items-start">
              <div>
                <BreedImage
                  images={result.breed.images}
                  alt={result.breed.name}
                  className="w-full aspect-square rounded-2xl shadow-medium"
                  breedName={result.breed.name}
                />
              </div>

              <div>
                <h1 className="text-4xl md:text-5xl font-bold mb-4">
                  {getBreedNamePT(result.breed.name)}
                </h1>
                
                <div className="mb-6">
                  <div className="flex justify-between mb-2">
                    <span className="text-lg font-semibold">Compatibilidade</span>
                    <span className="text-2xl font-bold text-primary">
                      {result.compatibilityScore}%
                    </span>
                  </div>
                  <Progress value={result.compatibilityScore} className="h-3" />
                  {result.compatibilityScore < 50 && (
                    <div className="mt-2 flex items-start gap-2 text-sm text-amber-700">
                      <AlertTriangle className="h-4 w-4 mt-0.5" />
                      <span>Recomendação com baixa confiança; considere ajustar suas preferências.</span>
                    </div>
                  )}
                </div>

                <p className="text-lg text-muted-foreground mb-6">
                  {result.breed.description || "Descrição não disponível para esta raça no momento."}
                </p>

                <div className="space-y-3 mb-8">
                  <h3 className="font-semibold text-lg">Por que esse match?</h3>
                  {result.matchReasons.map((reason, index) => (
                    <div key={`reason-${index}`} className="flex items-start gap-2">
                      <Heart className="w-5 h-5 text-secondary mt-0.5 flex-shrink-0" />
                      <span className="text-muted-foreground">{reason}</span>
                    </div>
                  ))}
                </div>

                <div className="flex flex-wrap gap-2 mb-8">
                  {result.breed.temperament.map((trait, index) => (
                    <Badge key={`trait-${index}`} variant="secondary" className="px-3 py-1">
                      {trait}
                    </Badge>
                  ))}
                </div>

                <div className="flex gap-3">
                  <Link to={`/breed/${encodeURIComponent(result.breed.name)}`} className="flex-1">
                    <Button className="w-full bg-gradient-hero">
                      Ver Detalhes Completos
                    </Button>
                  </Link>
                  <Button variant="outline" onClick={handleShare}>
                    <Share2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          </Card>

          {/* Top-k predições */}
          {result.topPredictions && result.topPredictions.length > 0 && (
            <div className="mb-8">
              <h2 className="text-2xl font-bold mb-4">Top recomendações do modelo</h2>
              <div className="grid md:grid-cols-3 gap-4">
                {result.topPredictions.map((pred, index) => (
                  <Card key={`top-${pred.name}-${index}`} className="p-4 flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">#{index + 1}</p>
                      <p className="font-semibold">{getBreedNamePT(pred.name)}</p>
                    </div>
                    <Badge variant="secondary">{pred.score}%</Badge>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {/* Similar Breeds */}
          <div className="mb-8">
            <h2 className="text-2xl font-bold mb-6">Raças Similares</h2>
            <div className="grid md:grid-cols-3 gap-6">
              {result.similarBreeds.slice(0, 3).map(({ breed, similarityScore }, index) => (
                <Link key={`similar-${breed.name || index}`} to={`/breed/${encodeURIComponent(breed.name)}`}>
                  <Card className="overflow-hidden hover:shadow-medium transition-all cursor-pointer group">
                    <div className="aspect-square overflow-hidden">
                      <BreedImage
                        images={breed.images}
                        alt={breed.name}
                        className="w-full h-full group-hover:scale-105 transition-transform duration-300"
                        breedName={breed.name}
                      />
                    </div>
                    <div className="p-6">
                      <div className="flex justify-between items-start mb-2">
                        <h3 className="text-xl font-bold">{getBreedNamePT(breed.name)}</h3>
                        <Badge variant="secondary">{similarityScore}%</Badge>
                      </div>
                      <p className="text-sm text-muted-foreground line-clamp-2">
                        {breed.description || "Detalhes adicionais não disponíveis."}
                      </p>
                    </div>
                  </Card>
                </Link>
              ))}
            </div>
          </div>

          {/* ML Explanation Card */}
          <Card className="p-6 bg-gradient-to-br from-primary/10 to-secondary/10">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <Brain className="w-6 h-6 text-primary" />
                  <h2 className="text-2xl font-bold">Como Funciona o Machine Learning?</h2>
                </div>
                <p className="text-muted-foreground">
                  Entenda como suas preferências são processadas e transformadas 
                  em recomendações. Veja o processo completo de Machine Learning por trás do DogMatch.
                </p>
              </div>
              <Link to="/ml-explanation">
                <Button className="ml-6 bg-gradient-hero">
                  <Brain className="mr-2 h-4 w-4" />
                  Ver Explicação
                </Button>
              </Link>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
