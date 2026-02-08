# report_generator.py

class XAIReportGenerator:

    def __init__(self, symbol):
        self.symbol = symbol

    def generate_report(
        self,
        price_explanation,
        sentiment_explanation,
        anomaly_explanation,
        decision
    ):

        report = []

        report.append("=====================================")
        report.append(f"RAPPORT XAI POUR : {self.symbol}")
        report.append("=====================================\n")

        report.append("DÉCISION FINALE :")
        report.append(f">>> {decision.upper()}\n")

        report.append("---- Explication de la prédiction ----")
        report.append(price_explanation)
        report.append("")

        report.append("---- Analyse du Sentiment ----")
        report.append(sentiment_explanation)
        report.append("")

        report.append("---- Analyse des Anomalies ----")
        report.append(anomaly_explanation)
        report.append("")

        report.append("Conclusion :")
        report.append(
            "Cette décision est basée sur une combinaison "
            "de prévisions ML, d'analyse de sentiment et de surveillance du marché."
        )

        return "\n".join(report)
