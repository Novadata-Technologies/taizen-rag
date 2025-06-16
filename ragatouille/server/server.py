import argparse
from typing import List, Dict, Optional
import os

import batched
import uvicorn
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from ragatouille import RAGPretrainedModel

app = FastAPI()

class DocumentMetadata(BaseModel):
    id: str
    project_id: str

class IndexRequest(BaseModel):
    """PyDantic model for the requests sent to the server.

    Parameters
    ----------
    input
        The input(s) to encode.
    metadata
        The metadata of the quotes.
    index_id
        The id of the index.
    """

    input: List[str]
    metadata: List[DocumentMetadata]
    index_id: str


class QueryRequest(BaseModel):
    """PyDantic model for the requests sent to the server.

    Parameters
    ----------
    input
        The input(s) to encode.
    index_id
        The id of the index
    """

    input: List[str]
    index_id: str
    k: Optional[int] = 5


class QueryResponse(BaseModel):
    """PyDantic model for the server answer to a call.

    Parameters
    ----------
    data
        pass
    model
        The name of the model used for encoding.
    """

    data: List[List[Dict]]
    model: str


def wrap_encode_function(model, **kwargs):
    def wrapped_encode(sentences):
        return model.encode(sentences, **kwargs)

    return wrapped_encode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FastAPI ColBERT serving server with specified host, port, and model."
    )
    parser.add_argument(
        "--index_name",
        type=str,
        help="Index name to load",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8002, help="Port to run the server on"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="jinaai/jina-colbert-v2",
        help="Model to serve, can be an HF model or a path to a model",
    )
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")

    return parser.parse_args()


args = parse_args()

if args.reload:
    os.environ["API_RELOAD"] = "true"
    print("API_RELOAD enabled by --reload")

# We need to load the model here so it is shared for every request
model = RAGPretrainedModel.from_index(args.index_name)
# We cannot create the function on the fly as the batching require to use the same function (memory address)
# TODO still can't do this, especially if we will be separating indices by project_id.
#model.search = batched.aio.dynamically(
#    wrap_encode_function(model, is_query=True)
#)
#model.encode_document = batched.aio.dynamically(
#    wrap_encode_function(model, is_query=False)
#)

router = APIRouter()

@router.post("/index")
async def index(request: IndexRequest):
    """API endpoint that encode the elements of an EmbeddingRequest and returns an EmbeddingResponse.

    Parameters
    ----------
    request
        The IndexRequest
    """
    try:
        model.add_to_index(
            request.input
        )
        return JSONResponse(status_code=201, content={"result": "ok"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """API endpoint that encode the elements of an EmbeddingRequest and returns an EmbeddingResponse.

    Parameters
    ----------
    request
        The QueryRequest
    """
    try:
        if request.k is None:
            k = 5
        else:
            k = request.k

        docs = model.search(request.input, k=k)
        if len(request.input) == 1:
            print("Only 1 query, wrapping...")
            docs = [docs]

        return QueryResponse(
            model="jinaai/jina-colbert-v2",
            data=docs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    module_path = "ragatouille.server.server:app"
    uvicorn.run(module_path, host=args.host, port=args.port, reload=reload)
