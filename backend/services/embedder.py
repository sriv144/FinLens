from sentence_transformers import SentenceTransformer

from backend.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL_NAME

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch encode a list of strings."""
    model = get_model()
    return model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()


def embed_query(text: str) -> list[float]:
    """Single query embedding."""
    model = get_model()
    return model.encode([text], normalize_embeddings=True)[0].tolist()
