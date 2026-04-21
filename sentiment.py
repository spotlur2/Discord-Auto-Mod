#Sentiment + Emotion + Sarcasm Detection Module
#pip install transformers torch

from transformers import pipeline

#load models
print("Loading models... (this may take a minute the first time)")

sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None
)

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

sarcasm_model = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-irony",
    top_k=None
)

print("All models loaded!")

def analyze_message(message: str, context: list[str] = None) -> dict:
    """
    Analyze a Discord message for sentiment, emotion, and sarcasm.

    Args:
        message: The message to analyze (string)
        context: Optional list of recent messages for context (from Sadd's module)

    Returns:
        Dictionary of scores for Ying's fusion module
    """

    #copntext
    if context:
        full_input = " [SEP] ".join(context + [message])
    else:
        full_input = message

    full_input = full_input[:512]

    #sentiment
    sentiment_raw = sentiment_model(full_input)[0]
    sentiment_scores = {item["label"].lower(): round(item["score"], 4) for item in sentiment_raw}

    #emotion
    emotion_raw = emotion_model(full_input)[0]
    emotion_scores = {item["label"].lower(): round(item["score"], 4) for item in emotion_raw}

    #sarcasm
    sarcasm_raw = sarcasm_model(full_input)[0]
    sarcasm_scores = {item["label"].lower(): round(item["score"], 4) for item in sarcasm_raw}

    #output
    result = {
        "message": message,
        "sentiment": sentiment_scores,
        "emotion": emotion_scores,
        "sarcasm": sarcasm_scores
    }

    return result


#FLAGGING
def get_risk_flags(analysis: dict) -> dict:

    flags = {
        "high_negative_sentiment": analysis["sentiment"].get("negative", 0) > 0.7,
        "high_anger":              analysis["emotion"].get("anger", 0) > 0.6,
        "high_disgust":            analysis["emotion"].get("disgust", 0) > 0.6,
        "likely_sarcastic":        analysis["sarcasm"].get("irony", 0) > 0.6,
    }
    return flags


"""
OUTPUT FORMAT
{
  "sentiment": {"positive": 0.85, "neutral": 0.10, "negative": 0.05},
  "emotion": {"anger": 0.6, "joy": 0.1, "sadness": 0.1, ...},
  "sarcasm": {"sarcastic": 0.72, "not_sarcastic": 0.28}
}
"""
