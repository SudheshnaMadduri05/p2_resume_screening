from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class ResumeMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def compute_batch_similarity(self, resumes, job_description):
        """
        Compute cosine similarity between a job description and
        multiple resumes using batch embeddings.
        """

        # Encode job description
        job_embedding = self.model.encode(
            job_description,
            convert_to_numpy=True
        )

        # Encode resumes in batches (FAST)
        resume_embeddings = self.model.encode(
            resumes,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Compute cosine similarity
        similarities = cosine_similarity(
            resume_embeddings,
            job_embedding.reshape(1, -1)
        ).flatten()

        return similarities
