"""
IMPROVED VERSION

This sentiment analyzer uses better prompt engineering:
1. Provides product-specific context to the LLM
2. Explains the intended purpose of each product
3. Includes more diverse examples, especially for products like TNT Sticks
4. Explicitly instructs the model to consider product context when determining sentiment

These improvements help the model correctly classify posts about products where
destructive features (explosions, crashes, etc.) are actually positive attributes.
"""

import requests
import json
import time
from zephyr_data import SOCIAL_MEDIA_POSTS


def create_sentiment_prompt(text, product):
    return f"""
    Analyze the sentiment expressed in this social media post regarding a Zephyr product named "{product}".
    Evaluate the sentiment in light of the product's intended purpose, as follows:
    - Rocket Skates: designed for speed and pursuit
    - TNT Sticks: engineered to explode and cause destruction (note: this is a desirable attribute)
    - Giant Magnet: intended to attract and draw objects
    - Anvil Drop Kit: built to deploy heavy objects onto targets
    - Bird Seed: formulated to attract birds
    
    Classify the sentiment as POSITIVE, NEGATIVE, or NEUTRAL, returning the classification in all uppercase letters.
    
    Examples:
    "Love these Zephyr Rocket Skates!" -> POSITIVE
    "Zephyr TNT blew up in my face!" -> NEGATIVE
    "The Giant Magnet works fine." -> NEUTRAL
    "TNT Sticks are awesome! Made a boom that echoed for miles." -> POSITIVE
    "Best TNT Sticks ever! Made a crater big enough to trap anyone." -> POSITIVE
    "Rocket Skates are too fast!" -> NEGATIVE
    "Anvil Drop Kit is flawless! Dropped right where I aimed." -> POSITIVE
    
    Post: "{text}"
    Sentiment:
    """


def analyze_sentiment(text, product):
    prompt = create_sentiment_prompt(text, product)
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma3:4b", "prompt": prompt, "stream": False},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        sentiment = result["response"].strip()
        sentiment_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
        return {"label": sentiment, "score": sentiment_map.get(sentiment, 0)}
    except Exception as e:
        print(f"Error: {e}")
        return {"label": "ERROR", "score": 0}


def process_batch(posts, batch_size=5):
    results = []
    for i in range(0, len(posts), batch_size):
        batch = posts[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        for post in batch:
            sentiment = analyze_sentiment(post["text"], post["product"])
            processed_post = post.copy()
            processed_post["sentiment"] = sentiment
            results.append(processed_post)
            time.sleep(0.5)  # Avoid overloading
        print(f"Batch {i // batch_size + 1} completed.")
    return results


def main():
    print("Starting Zephyr Sentiment Analysis with Gemma 3 4B...")
    results = process_batch(SOCIAL_MEDIA_POSTS)
    with open("sentiment_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Processed {len(results)} posts. Results saved to sentiment_results.json")


if __name__ == "__main__":
    main()
