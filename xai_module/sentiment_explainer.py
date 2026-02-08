# sentiment_explainer.py

class SentimentExplainer:

    def __init__(self, articles):
        """
        articles : liste de dictionnaires
        [
          {"title": "...", "score": 0.8, "source": "..."},
          {"title": "...", "score": -0.2, "source": "..."}
        ]
        """
        self.articles = articles

    def overall_sentiment(self):
        if not self.articles:
            return "neutre", 0

        avg = sum(a["score"] for a in self.articles) / len(self.articles)

        if avg > 0.2:
            return "positif", avg
        elif avg < -0.2:
            return "négatif", avg
        return "neutre", avg

    def explain(self):

        sentiment, avg = self.overall_sentiment()

        explanation = [
            f"Sentiment global détecté : {sentiment.upper()}",
            f"Score moyen : {round(avg, 2)}",
            "",
            "Articles ayant influencé la décision :"
        ]

        for art in self.articles:
            impact = "positif" if art["score"] > 0 else "négatif"
            explanation.append(
                f"- [{impact}] {art['title']} (source: {art['source']})"
            )

        return "\n".join(explanation)
