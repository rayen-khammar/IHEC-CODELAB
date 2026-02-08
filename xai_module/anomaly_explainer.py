# anomaly_explainer.py

class AnomalyExplainer:

    def __init__(self, anomalies):
        """
        anomalies : liste de signaux détectés
        [
          {"type": "volume", "value": 5.2},
          {"type": "price_jump", "value": 7.1}
        ]
        """
        self.anomalies = anomalies

    def has_anomalies(self):
        return len(self.anomalies) > 0

    def explain(self):

        if not self.anomalies:
            return "Aucune anomalie détectée. Le marché est stable."

        explanation = [
            "Des anomalies ont été détectées :"
        ]

        for anom in self.anomalies:

            if anom["type"] == "volume":
                explanation.append(
                    f"- Pic de volume inhabituel : {anom['value']}x la moyenne"
                )

            elif anom["type"] == "price_jump":
                explanation.append(
                    f"- Variation de prix anormale : {anom['value']}%"
                )

            elif anom["type"] == "volatility":
                explanation.append(
                    f"- Volatilité excessive détectée"
                )

        explanation.append(
            "Ces signaux suggèrent une prudence accrue."
        )

        return "\n".join(explanation)
