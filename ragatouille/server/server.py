import argparse
from typing import List, Dict, Optional
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
            if hasattr(RAG, 'cleanup'):
                print("🔧 Calling model cleanup method...")
                RAG.cleanup()
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

@router.post("/index/create")
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
        # RAGPretrainedModel.index will handle splitting and creating ColBERT index
        # Overwrite policy is handled by ColBERT (default "reuse")
        index_path = RAG.index(
            index_name=request.index_id,
            documents=request.input,
            document_ids=document_ids,
            document_metadatas=document_metadatas,
            # max_document_length, bsize, use_faiss_index can be exposed or configured as needed
        )
        print(f"Documents indexed for index_id: {request.index_id}. Index path: {index_path}")
        return JSONResponse(status_code=201, content={"result": "ok", "index_path": index_path})
    except ValueError as ve: # Catch specific errors like ID mismatches from RAGatouille
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error during indexing for index_id {request.index_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/add")
async def add_to_index(request: IndexRequest):
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
            model=RAG.get_model().base_pretrained_model_name_or_path, # Get base model name
            data=final_results
        )
    except FileNotFoundError as fnfe: # Specific error if index doesn't exist for querying
        print(f"Index not found for query: {request.index_id}. Error: {fnfe}")
        raise HTTPException(status_code=404, detail=f"Index '{request.index_id}' not found.")
    except Exception as e:
        print(f"Error during query for index_id {request.index_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    is_reload = os.getenv("API_RELOAD", "false").lower() == "true"
    module_path = "ragatouille.server.server:app" # Standard for uvicorn
    uvicorn.run(module_path, host=args.host, port=args.port, reload=is_reload)
