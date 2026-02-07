from openai import OpenAI

client = OpenAI()


def clean_text(text: str) -> str:
    """
    Basic text cleaning to reduce noise
    """
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def summarize_text(text: str, max_bullets: int = 5) -> str:
    """
    Summarize financial text into concise bullet points

    Input:
        text (str): raw financial text

    Output:
        str: summarized bullet points
    """

    cleaned_text = clean_text(text)

    prompt = f"""
    You are a financial analyst.

    Summarize the following text into {max_bullets} concise bullet points.
    Keep only factual, market-relevant information.
    Avoid opinions and speculation.

    Text:
    {cleaned_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()
