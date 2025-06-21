import argparse
from typing import List, Dict, Optional, Union
import os
from contextlib import asynccontextmanager

# import batched # Batched processing is currently commented out
import uvicorn
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from ragatouille import RAGPretrainedModel

# Global variable to store RAG model
RAG = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global RAG
    print("🚀 Starting up RAGatouille server...")
    try:
        print(f"📦 Loading RAGPretrainedModel with base model: {args.model}")
        print(f"📁 Index root set to: {args.index_root}")
        if args.initial_index_name:
            print(f"🔍 Attempting to pre-load initial index: {args.initial_index_name}")

        RAG = RAGPretrainedModel.from_pretrained(
            pretrained_model_name_or_path=args.model,
            index_root=args.index_root,
            initial_index_name=args.initial_index_name,
            experiment_name=args.experiment_name,
        )
        print("✅ RAGPretrainedModel loaded successfully.")
        print("🎯 Server is ready to accept requests!")
    except Exception as e:
        print(f"❌ Error loading RAGPretrainedModel: {e}")
        raise RuntimeError(f"Failed to initialize RAGPretrainedModel: {e}") from e

    yield

    # Shutdown
    print("🛑 Shutting down RAGatouille server...")
    cleanup_resources()
    print("✅ Server shutdown complete.")

app = FastAPI(lifespan=lifespan)

class DocumentMetadata(BaseModel):
    id: str # Unique ID for the original document
    project_id: str # Example metadata field

class IndexRequest(BaseModel):
    """PyDantic model for the requests sent to the /index endpoint.

    Parameters
    ----------
    input
        A list of document texts to be indexed.
    metadata
        A list of metadata objects, one for each document in `input`.
    index_id
        The identifier for the index to be created or updated.
    """

    input: List[str]
    metadata: List[DocumentMetadata]
    index_id: str


class DeleteDocRequest(BaseModel):
    """PyDantic model for the server answer to a delete document request.

    Parameters
    ----------
    documents
        Dictionary mapping index IDs to lists of document IDs.
    """

    documents: Dict[str, List[str]]


class QueryRequest(BaseModel):
    """PyDantic model for the requests sent to the /query endpoint.

    Parameters
    ----------
    input
        A list of query strings.
    index_id
        The identifier of the index to query.
    k
        Optional number of results to return per query.
    """

    input: List[str]
    index_id: str
    k: Optional[int] = 5


class QueryResponse(BaseModel):
    """PyDantic model for the server answer to a /query call.

    Parameters
    ----------
    data
        List of search results. For multiple input queries, this will be a list of lists.
    model
        The name of the base model used for encoding.
    """

    data: List[List[Dict]] # Assuming search always returns a list of lists for multiple queries
    model: str


class ListIndexResponse(BaseModel):
    """PyDantic model for the server answer to a /index/list call.

    Parameters
    ----------
    data
        List of index names.
    """

    indexes: Dict[str, Dict[str, Union[int, bool, None]]]


class DeleteIndexRequest(BaseModel):
    """PyDantic model for the requests sent to the /index/{index_id} DELETE endpoint.

    Parameters
    ----------
    index_id
        The identifier of the index to delete.
    """

    index_id: str


class DeleteIndexResponse(BaseModel):
    """PyDantic model for the server answer to a DELETE index request.

    Parameters
    ----------
    result
        Status of the deletion operation.
    deleted_index
        The name of the deleted index.
    """

    result: str
    deleted_index: str


# Batched processing currently not used with the single model refactor
# def wrap_encode_function(model, **kwargs):
#     def wrapped_encode(sentences):
#         return model.encode(sentences, **kwargs)
#     return wrapped_encode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FastAPI RAGatouille server with specified host, port, base model, and index root."
    )
    parser.add_argument(
        "--model",
        type=str,
        # default="jinaai/jina-colbert-v2", # This seems awful for now
        default="colbert-ir/colbertv2.0",
        help="Base model to serve, can be an HF model name or a path to a local ColBERT checkpoint.",
    )
    parser.add_argument(
        "--index-root",
        dest="index_root",
        type=str,
        default="./ragatouille/colbert/indexes", # Default root directory for all indices
        help="Root directory where all named indices will be stored and loaded from.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on."
    )
    parser.add_argument(
        "--port", type=int, default=8002, help="Port to run the server on."
    )
    parser.add_argument(
        "--experiment-name",
        dest="experiment_name",
        type=str,
        default="colbert",
        help="Experiment name used in structuring index paths under index_root.",
    )
    parser.add_argument(
        "--initial-index-name",
        dest="initial_index_name",
        type=str,
        default=None,
        help="Optional: Name of an index within index_root to pre-load at startup."
    )
    parser.add_argument("--reload", action="store_true", help="Enable hot reload for development.")

    return parser.parse_args()


args = parse_args()

if args.reload:
    os.environ["API_RELOAD"] = "true"
    print("API_RELOAD enabled by --reload flag.")

def cleanup_resources():
    """Clean up resources on shutdown"""
    global RAG
    if RAG is not None:
        print("🧹 Cleaning up RAGPretrainedModel resources...")
        try:
            # If the model has any cleanup methods, call them here
            # Note: RAGPretrainedModel doesn't have a cleanup method currently
            # but we can add custom cleanup logic here if needed
            # Force garbage collection to help with cleanup
            print("🗑️  Running garbage collection...")
            import gc
            gc.collect()
            # Clear the model reference
            RAG = None
            print("✅ RAGPretrainedModel cleanup completed.")
        except Exception as e:
            print(f"⚠️  Error during RAGPretrainedModel cleanup: {e}")
    else:
        print("ℹ️  No RAGPretrainedModel to clean up.")


router = APIRouter()

@router.post("/index")
async def create_index(request: IndexRequest):
    """
    API endpoint to index a list of documents.
    If the index_id does not exist, a new index will be created.
    If it exists, behavior depends on the underlying ColBERT index overwrite policy (default "reuse").
    """
    try:
        if not request.input:
            raise HTTPException(status_code=400, detail="Input documents list cannot be empty.")
        if len(request.input) != len(request.metadata):
            raise HTTPException(status_code=400, detail="Length of input documents and metadata must match.")

        document_ids = [meta.id for meta in request.metadata]
        # Ensure document_ids are unique as RAGPretrainedModel expects this
        if len(document_ids) != len(set(document_ids)):
            raise HTTPException(status_code=400, detail="Document IDs in metadata must be unique.")

        document_metadatas = [{"project_id": meta.project_id} for meta in request.metadata]

        print(f"Indexing documents for index_id: {request.index_id}...")

        if RAG is None:
            raise HTTPException(status_code=500, detail="RAG model not initialized")

        # RAGPretrainedModel.index will handle splitting and creating ColBERT index
        # Overwrite policy is handled by ColBERT (default "reuse")
        index_path = RAG.index(
            index_name=request.index_id,
            documents=request.input,
            document_ids=document_ids,
            document_metadatas=document_metadatas,
            overwrite_index="force_silent_overwrite"
            # max_document_length, bsize, use_faiss_index can be exposed or configured as needed
        )
        print(f"Documents indexed for index_id: {request.index_id}. Index path: {index_path}")
        return JSONResponse(status_code=201, content={"result": "ok", "index_path": index_path})
    except ValueError as ve: # Catch specific errors like ID mismatches from RAGatouille
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error during indexing for index_id {request.index_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/index/doc")
#TODO alter this so it takes a dict, mapping index ids to documents
async def add_to_index(request: IndexRequest):
    """
    API endpoint to add a document to an index.
    If the index_id does not exist, a new index will be created.
    If it exists, behavior depends on the underlying ColBERT index overwrite policy (default "reuse").
    """
    try:
        if not request.input:
            raise HTTPException(status_code=400, detail="Input documents list cannot be empty.")
        if len(request.input) != len(request.metadata):
            raise HTTPException(status_code=400, detail="Length of input documents and metadata must match.")

        document_ids = [meta.id for meta in request.metadata]
        # Ensure document_ids are unique as RAGPretrainedModel expects this
        if len(document_ids) != len(set(document_ids)):
            raise HTTPException(status_code=400, detail="Document IDs in metadata must be unique.")

        document_metadatas = [{"project_id": meta.project_id} for meta in request.metadata]

        print(f"Indexing documents for index_id: {request.index_id}...")

        if RAG is None:
            raise HTTPException(status_code=500, detail="RAG model not initialized")

        RAG.add_to_index(
            index_name=request.index_id,
            new_documents=request.input,
            new_document_ids=document_ids,
            new_document_metadatas=document_metadatas,
        )
        return JSONResponse(status_code=201, content={"result": "ok"})
    except ValueError as ve: # Catch specific errors like ID mismatches from RAGatouille
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error during indexing for index_id {request.index_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/index/doc")
async def delete_from_index(request: DeleteDocRequest):
    """
    API endpoint to delete a series of documents from a series of indexes.
    """
    def format_response(status_code: int, success: dict, failure: dict, error: str):
        return JSONResponse(
            status_code=status_code,
            content={
                "success": success,
                "failure": failure,
                "error": error
            }
        )
    try:
        if RAG is None:
            raise HTTPException(status_code=500, detail="RAG model not initialized")

        success = {}
        failure = request.documents.copy()
        for index_id, document_ids in request.documents.items():
            RAG.delete_from_index(
                index_name=index_id,
                document_ids=document_ids
            )
            success[index_id] = document_ids
            del failure[index_id]
        return format_response(200, success, failure, "")
    except KeyError as ke:
        return format_response(404, success, failure, str(ke))
    except FileNotFoundError as fe:
        return format_response(404, success, failure, str(fe))
    except Exception as e:
        return format_response(500, success, failure, str(e))

@router.post("/query", response_model=QueryResponse)
async def query_index(request: QueryRequest):
    """
    API endpoint to query an index.
    """
    try:
        if not request.input:
            raise HTTPException(status_code=400, detail="Input query list cannot be empty.")

        k_val = request.k if request.k is not None else 5 # Default k if not provided

        print(f"Querying index_id: {request.index_id} with k={k_val}...")

        if RAG is None:
            raise HTTPException(status_code=500, detail="RAG model not initialized")

        # RAGPretrainedModel.search handles loading the correct index context
        search_results = RAG.search(
            index_name=request.index_id,
            query=request.input,
            k=k_val
        )

        # Ensure results are always a list of lists, even for a single query
        if len(request.input) == 1 and not isinstance(search_results[0], list):
            final_results = [search_results]
        else:
            final_results = search_results

        print(f"Query successful for index_id: {request.index_id}.")
        return QueryResponse(
            model=str(RAG.get_model().base_pretrained_model_name_or_path), # Get base model name
            data=final_results
        )
    except FileNotFoundError as fnfe: # Specific error if index doesn't exist for querying
        print(f"Index not found for query: {request.index_id}. Error: {fnfe}")
        raise HTTPException(status_code=404, detail=f"Index '{request.index_id}' not found.")
    except Exception as e:
        print(f"Error during query for index_id {request.index_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/index/list", response_model=ListIndexResponse)
async def list_indexes():
    """
    API endpoint to list all available indexes.
    """
    try:
        if RAG is None:
            raise HTTPException(status_code=500, detail="RAG model not initialized")

        return ListIndexResponse(indexes=RAG.get_available_indexes())
    except Exception as e:
        print(f"Error during index listing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/index/{index_id}", response_model=DeleteIndexResponse)
async def delete_index(index_id: str):
    """
    API endpoint to delete an entire index.
    """
    try:
        if RAG is None:
            raise HTTPException(status_code=500, detail="RAG model not initialized")

        print(f"Attempting to delete index: {index_id}")

        # Check if index exists first
        available_indexes = RAG.get_available_indexes()
        if index_id not in available_indexes:
            raise HTTPException(status_code=404, detail=f"Index '{index_id}' not found.")

        # Delete the index
        RAG.delete_index(index_name=index_id)

        print(f"Successfully deleted index: {index_id}")
        return DeleteIndexResponse(result="ok", deleted_index=index_id)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except FileNotFoundError as fnfe:
        print(f"Index not found for deletion: {index_id}. Error: {fnfe}")
        raise HTTPException(status_code=404, detail=f"Index '{index_id}' not found.")
    except Exception as e:
        print(f"Error during index deletion for index_id {index_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    is_reload = os.getenv("API_RELOAD", "false").lower() == "true"
    module_path = "ragatouille.server.server:app" # Standard for uvicorn
    uvicorn.run(module_path, host=args.host, port=args.port, reload=is_reload)
