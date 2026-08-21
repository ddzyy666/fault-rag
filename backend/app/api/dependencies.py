from typing import Annotated

from fastapi import Depends

from app.services.embedding import EmbeddingProvider, get_embedding_provider
from app.services.vector_store import QdrantVectorStore, get_vector_store

EmbeddingDependency = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]
VectorStoreDependency = Annotated[QdrantVectorStore, Depends(get_vector_store)]
