import pandas as pd
import sklearn
import openai
from openai import OpenAI

# Data
Sentences_AllAgree = pd.read_csv('Sentences_AllAgree.txt',encoding = "ISO-8859-1", names=['text','sentiment'], delimiter= '@')

# Enter your OpenAI API key here
client = OpenAI(
    api_key='OpenAI API key',
)

def sentiment(text: str, model) -> int:
    """Compute the sentiment for the text.

    text: string containing the text
    model: string characterizing the openai model to use
    """

    # build the prompt
    prompt = f"""Label the sentiment of the given text. The answer should be exact 'positive', 'neutral' or 'negative'.

    Text: {text}
    The answer is
    """

    response = client.completions.create(model=model, prompt=prompt, max_tokens=50)
    generated_text = response.choices[0].text

    std_gen_text = generated_text.lower().strip()

    if std_gen_text == "positive":
        return "positive"
    elif std_gen_text == "negative":  # elseif
        return "negative"
    elif std_gen_text == "neutral":
        return "neutral"
    else:
        print("Unrecognized output")
        return pd.NA

# GPT 3.5
values_3 = Sentences_AllAgree['text'].apply(sentiment, model="gpt-3.5-turbo-instruct")
Sentences_AllAgree["predicted_sentiment_3.5"] = values_3

# GPT 4.0
values_4 = Sentences_AllAgree['text'].apply(sentiment, model="gpt-4")
Sentences_AllAgree["predicted_sentiment_4.0"] = values_4

Sentences_AllAgree.to_csv('result.csv')
