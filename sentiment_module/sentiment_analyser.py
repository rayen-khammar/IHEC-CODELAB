import json
from summarizer import summarize_text
from config import SENTIMENT_THRESHOLDS
from openai import OpenAI

client = OpenAI()


def interpret_sentiment(score: float) -> str:
    """
    Convert numerical score into label
    """
    if score <= SENTIMENT_THRESHOLDS["negative"]:
        return "NEGATIVE"
    elif score >= SENTIMENT_THRESHOLDS["positive"]:
        return "POSITIVE"
    else:
        return "NEUTRAL"


def analyze_sentiment(file_path: str) -> dict:
    """
    Main sentiment analysis pipeline

    Input:
        file_path (str): path to .txt file

    Output:
        dict with summary, sentiment score, label, confidence
    """

    # 1. Read file
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 2. Summarize text
    summary = summarize_text(raw_text)

    # 3. Ask OpenAI for sentiment score
    prompt = f"""
    Analyze the sentiment of the following summarized financial text.

    Return ONLY a valid JSON with this exact format:
    {{
        "sentiment_score": number between -1 and 1,
        "confidence": number between 0 and 1
    }}

    Text:
    {summary}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    # 4. Parse response
    sentiment_data = json.loads(response.choices[0].message.content)

    score = float(sentiment_data["sentiment_score"])
    confidence = float(sentiment_data["confidence"])

    # 5. Interpret score
    label = interpret_sentiment(score)

    # 6. Final output
    return {
        "summary": summary,
        "sentiment_score": score,
        "sentiment_label": label,
        "confidence": confidence
    }


if __name__ == "__main__":
    result = analyze_sentiment("example/input.txt")
    print(json.dumps(result, indent=2))
