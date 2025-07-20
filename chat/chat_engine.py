"""
Chat Engine for Zephyr Analytics

This module provides functionality to search and retrieve relevant comments
based on semantic similarity using FAISS vector database.
"""

import faiss
import numpy as np
import json
import pickle
import os
from datetime import datetime


try:
    from chat.embedding_model import EmbeddingModel
except ImportError:
 
    try:
        from embedding_model import EmbeddingModel
    except ImportError:
        print("Error: Could not import EmbeddingModel!")


class ZephyrChatEngine:
    """
    Chat engine that uses FAISS to find semantically similar comments
    and generates responses based on those comments.
    """

    def __init__(self, data_path="../data/sentiment_results.json"):
        """
        Initialize the chat engine.
        """

        self.embedding_model = EmbeddingModel()
        self.index = None

        if not os.path.exists(data_path):
            print("Error: sentiment_results.json not found!")
            print("Please run the sentiment analysis step first.")
            return
        
        with open(data_path, "r") as f:
            self.comments_data = json.load(f)
            self.comment_texts = [post["text"] for post in self.comments_data]


        faiss_path = os.path.join(os.path.dirname(data_path), "faiss_index.pkl")

    
        if os.path.exists(faiss_path):
            self.load_index(faiss_path)
        else:
            # We need to build the index
            print("Building FAISS index...")
            self.build_index(faiss_path)

        
    def build_index(self, index_path="../data/faiss_index.pkl"):
        """
        Build and save the FAISS index for fast similarity search.
        """

        try:
            # Generate embeddings using the embedding_model
            print(f"Generating embeddings for {len(self.comment_texts)} comments...")
            embeddings = self.embedding_model.encode(self.comment_texts)

            # Create and populate FAISS index
            dimension = embeddings.shape[1]
            print(f"Creating FAISS index with dimension {dimension}...")

            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype("float32"))

            # Save index for future use
            print(f"Saving FAISS index to {index_path}...")

            with open(index_path, "wb") as f:
                pickle.dump({"index": faiss.serialize_index(self.index)}, f)

            # Clean up memory, we don't need the comment text anymore
            del self.comment_texts

            return self.index
        except Exception as e:
            print(f"Error in build_index: {e}")
            return None

    def load_index(self, load_path="../data/faiss_index.pkl"):
        """
        Load a previously saved FAISS index.
        """

        if(os.path.exists(load_path)):
            with open(load_path, "rb") as f:
                data = pickle.load(f)
                self.index = faiss.deserialize_index(data["index"])
            return self.index
        else:
            raise FileNotFoundError(
                f"Could not find faiss_index.pkl in the expected location: {load_path}"
            )

    def search_similar_comments(self, query, k=5, filters=None):
        """
        Retrieve comments semantically similar to the query, with optional filtering.

        Args:
            query (str): Text to find similar comments for.
            k (int): Number of similar comments to return.
            filters (dict): Optional criteria to refine results:
                - product (list): Products to include.
                - sentiment (list): Sentiment labels to include.
                - commenter (list): Commenter names to include.
                - date_range (tuple): Start and end dates in 'YYYY-MM-DD' format.

        Returns:
            list: Comment dictionaries matching the query and filters.
        """
        # Convert query to vector
        query_vector = self.embedding_model.encode([query])

        # Get 5x results to filter down
        search_k = min(len(self.comments_data), k * 5)

        distances, indices = self.index.search(
            np.array(query_vector).astype("float32"), search_k
        )
        indices = indices[0]  # FAISS returns a 2D array, we need the first row

        # Get the initial results
        results = [self.comments_data[idx] for idx in indices]

        # Apply filters if provided
        if filters:
            filtered_results = []
            for comment in results:
                # Check product filter
                if (
                    filters.get("product")
                    and comment["product"] not in filters["product"]
                ):
                    continue

                # Check sentiment filter
                if (
                    filters.get("sentiment")
                    and comment["sentiment"]["label"] not in filters["sentiment"]
                ):
                    continue

                # Check commenter filter
                if (
                    filters.get("commenter")
                    and comment["commenter"] not in filters["commenter"]
                ):
                    continue

                # Check date range filter
                if filters.get("date_range"):
                    start_date, end_date = filters["date_range"]
                    comment_date = datetime.strptime(comment["date"], "%Y-%m-%d")
                    if start_date:
                        start = datetime.strptime(start_date, "%Y-%m-%d")
                        if comment_date < start:
                            continue
                    if end_date:
                        end = datetime.strptime(end_date, "%Y-%m-%d")
                        if comment_date > end:
                            continue

                filtered_results.append(comment)

                # Stop once we have enough results
                if len(filtered_results) >= k:
                    break

            return filtered_results[:k]

        # No filters, just return top k
        return results[:k]


    def generate_chat_prompt(self, query, similar_comments):
        """
        Construct a prompt for the LLM incorporating the user's query and context 
        from similar comments.

        Args:
            query (str): The question posed by the user.
            similar_comments (list): A collection of comment dictionaries to provide 
                                     context.

        Returns:
            str: The formulated prompt for submission to the LLM.
        """
        # Include sentiment information in the context
        context = "\n".join(
            [
                f"Comment about {c['product']} (Sentiment: {c['sentiment']['label']}):" + 
                    f"\"{c['text']}\" - by {c['commenter']} on {c['date']}"
                for c in similar_comments
            ]
        )

        prompt = f"""
        You are tasked with analyzing customer comments regarding Zephyr Innovations products.
        Interpret these comments in light of each product's intended purpose:
        - Rocket Skates: designed for speed and pursuit
        - TNT Sticks: engineered to explode and cause destruction 
          (note: this is a desirable attribute)
        - Giant Magnet: intended to attract and draw objects
        - Anvil Drop Kit: built to deploy heavy objects onto targets
        - Bird Seed: formulated to attract birds
    
        Below are relevant comments with their sentiment classifications: {context}
    
        User question: {query}
    
        Respond to the user's question based on these comments. 
        Provide specific references to the comments in your response.
        Include observations on sentiment patterns if pertinent to the question.
        """

        return prompt


    def chat(self, query, k=5, filters=None):
        """
        Generate an AI response to a query based on semantically 
        similar comments.

        Args:
            query (str): The user's question.
            k (int): Number of similar comments to consider.
            filters (dict): Optional criteria to refine the comment search.

        Returns:
            dict: A dictionary containing the response, relevant comments, 
                  and the prompt utilized.
        """
        # Step 1: Find comments similar to the query
        similar_comments = self.search_similar_comments(query, k, filters)
    
        # Step 2: Generate a prompt incorporating these comments
        prompt = self.generate_chat_prompt(query, similar_comments)

        # Send to Ollama API
        import requests

        try:
            # Step 3: Verify Ollama availability with a health check
            try:
                health_check = requests.get("http://localhost:11434/api/version", 
                                            timeout=2)
                health_check.raise_for_status()
            except requests.exceptions.RequestException:
                # Return formatted error message if Ollama is unavailable
                return {
                    "answer": "The Ollama service is not currently operational.",
                    "relevant_comments": similar_comments,
                    "prompt": prompt,
                    "error": "Ollama API not available",
                }

            # Step 4: Submit the prompt to Ollama
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "gemma3:4b", "prompt": prompt, "stream": False},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            
            # Step 5: Process and return the response
            return {
                "answer": result["response"],
                "relevant_comments": similar_comments,
                "prompt": prompt,
            }
        except requests.exceptions.Timeout:
            # Handle timeout errors with helpful message
            print("Request to Ollama API exceeded time limit.")
            return {
                "answer": """
                ⏱️ **Request Timed Out**
                
                The request to the Ollama API exceeded the allotted time. Possible causes include:
                
                1. The model is still initializing or processing a prior request
                2. The prompt is excessively large (consider reducing the number of comments)
                3. System resources are under strain
                
                Please retry with fewer comments or wait briefly before resubmitting.
                """,
                "relevant_comments": similar_comments,
                "prompt": prompt,
                "error": "Request timed out",
            }
        except Exception as e:
            # Handle unexpected errors
            print(f"Error encountered: {e}")
            return {
                "answer": f"""
                ❌ **Error Generating Response**
                
                {str(e)}
                
                Please verify:
                - Ollama is operational with the gemma3:4b model
                - Network connectivity is stable
                - Sufficient system resources are available
                """,
                "relevant_comments": similar_comments,
                "prompt": prompt,
                "error": str(e),
            }

    def get_unique_values(self):
        """
        Get unique values for filter dropdowns. This is for the web interface.

        Returns:
            dict: Dictionary containing unique products, commenters, sentiments, and date range.
        """
        products = sorted(list(set(c["product"] for c in self.comments_data)))
        commenters = sorted(list(set(c["commenter"] for c in self.comments_data)))
        sentiments = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

        # Get min and max dates
        dates = [c["date"] for c in self.comments_data]
        min_date = min(dates)
        max_date = max(dates)

        return {
            "products": products,
            "commenters": commenters,
            "sentiments": sentiments,
            "date_range": (min_date, max_date),
        }
