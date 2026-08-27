from typing import Annotated

from fastapi import Depends

from app.services.embedding import EmbeddingProvider, get_embedding_provider
from app.services.llm import LLMProvider, get_llm_provider
from app.services.reranker import RerankerProvider, get_reranker_provider
from app.services.vector_store import QdrantVectorStore, get_vector_store

EmbeddingDependency = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]
VectorStoreDependency = Annotated[QdrantVectorStore, Depends(get_vector_store)]
LLMDependency = Annotated[LLMProvider, Depends(get_llm_provider)]
RerankerDependency = Annotated[RerankerProvider, Depends(get_reranker_provider)]
