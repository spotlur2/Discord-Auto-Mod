""" 
    Feature Fusion
        1) It combines outputs from multiple models into 
        a unified feature represention for each message.
        2) It prepares the data for the scoring and making
        a decision to the user.
"""

# Helper Functions

def clamp_to_01(value):
    """
    The function forces a value to stay between 0 and 1
    to ensure a consistency before combining since
    different models might have different range of outputs.
    """
    
    try:
        # convert value into numbers
        value = float(value)
    except (TypeError, ValueError):
        # default = 0
        return 0.0
        
    # if nothing goes wrong, force value stay between 0 and 1
    return max(0.0, min(value, 1.0))
    
def normalize_sentiment(sentiment):
    """
    The function converts score from range [-1, 1] to [0, 1]
    because sentiment model might output values from -1 to 1
    """
    
    try:
        # convert sentiment into numbers
        sentiment = float(sentiment)
    except (TypeError, ValueError):
        # default = neutral (0.5)
        return 0.5
        
    # if nothing goes wrong, convert it
    return (sentiment + 1.0) / 2.0
    
# Fusion Functions

def fuse_features(message, model_outputs):
    """
    The function takes
        - the message
        - outputs from all models
    then it combines everything into one dictionary of features.
    
    Parameters:
        message (str): the Discord message
        model_outputs (dict): outputs from different models
        
    Returns:
        dict: one fused feature representation
    """
    
    # get data from each model
    # if a model is missing, use empty dictionary
    toxicity_data = model_outputs.get("toxicity_model", {})
    sentiment_data = model_outputs.get("sentiment_model", {})
    sarcasm_data = model_outputs.get("sarcasm_model", {})
    spam_data = model_outputs.get("spam_model", {})
    
    # combine every feature into one
    fused = {
        
        "message": message,
        
        # Toxicity features: from toxicity classification model
        "toxicity": clamp_to_01(toxicity_data.get("toxicity", 0.0)),
        "insult": clamp_to_01(toxicity_data.get("insult", 0.0)),
        "threat": clamp_to_01(toxicity_data.get("threat", 0.0)),
        
        # Sentiment & Emotion features: it's normalized into 0-1 range
        "sentiment": normalize_sentiment(sentiment_data.get("sentiment", 0.0)),
        "anger": clamp_to_01(sentiment_data.get("anger", 0.0)),
        
        # Sarcasm feature
        "sarcasm": clamp_to_01(sarcasm_data.get("sarcasm", 0.0)),
        
        # Spam & Behavioral features
        "spam": clamp_to_01(spam_data.get("spam", 0.0)),
        "repetition": clamp_to_01(spam_data.get("repetition", 0.0)),
        "url_risk": clamp_to_01(spam_data.get("url_risk", 0.0)),
    }
    
    return fused
    
# Batch Fusion (for multiple messages)

def fuse_multiple(messages_with_outputs):
    """
    The function processes multiple messages in one pass.
        The loops go through a list and combines everything
    
    Parameters:
        messages_with_outputs (list): 
            each item contains message and model outputs
    """
    
    fused_outputs = []
    
    for i in messages_with_outputs:
        # get message
        # if it's missing, set empty as a default
        message = i.get("message", "")
        
        # get model outputs
        # if it's missing, set empty as a default
        outputs = i.get("outputs", {})
        
        # fuse features for this message
        fused = fuse_features(message, outputs)
        
        # store the result in the list
        fused_outputs.append(fused)
        
    return fused_outputs
    
    
#### Example for testing ####
#############################

if __name__ == "__main__":
    
    print("----- Feature Fusion Test -----\n")
    
    # example input data (from other models)
    test_data = [ 
        {
            "message": "you are so bad at this lol",
            "outputs": {
                "toxicity_model": {
                    "toxicity": 0.83,
                    "insult": 0.72,
                    "threat": 0.10
                },
                "sentiment_model": {
                    "sentiment": -0.65,
                    "anger": 0.70
                },
                "sarcasm_model": {
                    "sarcasm": 0.25
                },
                "spam_model": {
                    "spam": 0.58,
                    "repetition": 0.20,
                    "url_risk": 0.90
                }
            }
        },
        {
            "message": "Join my server now! discord.gg/test_data",
            "outputs": {
                "toxicity_model": {
                    "toxicity": 0.03,
                    "insult": 0.01,
                    "threat": 0.00
                },
                "sentiment_model": {
                    "sentiment": 0.1,
                    "anger": 0.03
                },
                "sarcasm_model": {
                    "sarcasm": 0.01
                },
                "spam_model": {
                    "spam": 0.90,
                    "repetition": 0.60,
                    "url_risk": 0.95
                }
            }
        }
    ]

    # run fusion for all messages
    fused_outputs = fuse_multiple(test_data)

    # print the results
    print("Fused result:\n")
    for i, output in enumerate(fused_outputs):
        print(f"--- Message {i+1} ---")
        for key, value in output.items():
            print(f"{key}: {value}")
        print()