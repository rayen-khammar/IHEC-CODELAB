# price_explainer.py

class PriceExplainer:

    def __init__(self, model_features):
        """
        model_features : dictionnaire contenant l'importance des features
        Exemple :
        {
          "RSI": 0.35,
          "Volume": 0.25,
          "Moving Average": 0.20,
          "Sentiment": 0.20
        }
        """
        self.features = model_features

    def explain_prediction(self, prediction):
        explanation = []

        direction = "hausse" if prediction > 0 else "baisse"

        explanation.append(f"Le modèle prévoit une {direction} de {abs(prediction)}%.")

        sorted_features = sorted(
            self.features.items(),
            key=lambda x: x[1],
            reverse=True
        )

        explanation.append("Les principaux facteurs influençant cette décision sont :")

        for feature, weight in sorted_features:
            percent = round(weight * 100, 1)
            explanation.append(f"- {feature} : contribution de {percent}%")

        return "\n".join(explanation)

    def confidence_score(self):
        return min(95, int(sum(self.features.values()) * 100))
