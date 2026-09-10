"""Rebuild all chunked document indexes with the currently configured vector schema."""

import asyncio

from app.db.database import async_session_factory, engine
from app.models.document import Document, DocumentStatus
from app.services.document_indexing import index_document
from app.services.embedding import get_embedding_provider
from app.services.sparse_embedding import get_sparse_embedding_provider
from app.services.vector_store import get_vector_store
from sqlalchemy import select


async def main() -> None:
    embedding = get_embedding_provider()
    sparse_embedding = get_sparse_embedding_provider()
    vector_store = get_vector_store()
    try:
        async with async_session_factory() as session:
            documents = list(
                (
                    await session.scalars(
                        select(Document)
                        .where(
                            Document.status.in_([DocumentStatus.CHUNKED, DocumentStatus.INDEXED])
                        )
                        .order_by(Document.created_at)
                    )
                ).all()
            )
            print(f"documents_to_index={len(documents)}", flush=True)
            for document in documents:
                result = await index_document(
                    session,
                    document,
                    embedding,
                    sparse_embedding,
                    vector_store,
                )
                print(
                    f"indexed={document.filename} vectors={result.vector_count} "
                    f"elapsed_ms={result.elapsed_ms}",
                    flush=True,
                )
    finally:
        vector_store.client.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
