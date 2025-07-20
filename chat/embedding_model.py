"""
Embedding Model for Zephyr Chat

This module provides functionality to convert text to vector embeddings
using the sentence-transformers library.
"""

# Import warning suppression before anything else
import warnings

# Suppress PyTorch class registration warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*torch.tensor results are registered as constants.*",
)

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Handles text-to-vector conversion using a pre-trained sentence transformer model."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        Args:
            model_name (str): Name of the sentence-transformers model to use.
                              Default is 'all-MiniLM-L6-v2', a lightweight model
                              with good performance.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    
    
    def encode(self, texts):
        """
        Convert text(s) to vector embeddings.

        Args:
            texts (str or list): A single text string or a list of text strings to encode.

        Returns:
            numpy.ndarray: Vector embeddings of the input text(s).
        """
        return self.model.encode(texts)

    
    
    def get_dimension(self):
        """
        Get the dimension of the embedding vectors.

        Returns:
            int: Dimension of the embedding vectors.
        """
        return self.model.get_sentence_embedding_dimension()
