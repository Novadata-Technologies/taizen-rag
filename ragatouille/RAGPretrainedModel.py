import os
from pathlib import Path
from typing import Callable, List, Optional, Union, Dict, Any, Tuple, Literal
from uuid import uuid4

from ragatouille.models.colbert import ColBERT
from ragatouille.data.corpus_processor import CorpusProcessor
from ragatouille.data.preprocessors import llama_index_sentence_splitter
from colbert.infra import ColBERTConfig


class RAGPretrainedModel:
    """
    Wrapper class for ColBERT models, supporting multiple indices
    managed by a single underlying neural network model.
    """

    def __init__(
        self,
        colbert_model: ColBERT,
        corpus_processor: CorpusProcessor,
        verbose: int = 1,
    ):
        """
        Initializes the RAGPretrainedModel.

        Parameters:
            colbert_model: An initialized instance of the ColBERT model.
            corpus_processor: An initialized instance of the CorpusProcessor.
            verbose: Verbosity level.
        """
        self.model: ColBERT = colbert_model
        self.corpus_processor: CorpusProcessor = corpus_processor
        self.verbose: int = verbose

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        index_root: Optional[str] = None,
        initial_index_name: Optional[str] = None,
        n_gpu: int = -1,
        verbose: int = 1,
        experiment_name: str = "colbert",
        document_splitter_fn: Optional[Callable] = llama_index_sentence_splitter,
        preprocessing_fns: Optional[Union[Callable, List[Callable]]] = None,
        **colbert_kwargs: Any,
    ):
        """
        Load a ColBERT model from a pretrained path or Hugging Face model name.
        This instance can then manage multiple indices under the specified index_root.

        Parameters:
            pretrained_model_name_or_path: Path to a local ColBERT checkpoint or HF model name.
            index_root: The root directory where all named indices will be stored/loaded from.
                        Defaults to '.ragatouille/' within the ColBERT class if None.
            initial_index_name: Optionally, the name of an index to load immediately.
            n_gpu: Number of GPUs to use. -1 for all available.
            verbose: Verbosity level.
            experiment_name: Name of the experiment, used in structuring the index path.
            document_splitter_fn: Function to split documents into passages.
            preprocessing_fns: Functions to preprocess passages.
            **colbert_kwargs: Additional keyword arguments for ColBERTConfig for new indices.
        """
        colbert_model_instance = ColBERT(
            pretrained_model_name_or_path=str(pretrained_model_name_or_path),
            index_root=index_root,
            initial_index_name=initial_index_name,
            n_gpu=n_gpu,
            verbose=verbose,
            experiment_name=experiment_name,
            **colbert_kwargs,
        )
        corpus_processor_instance = CorpusProcessor(
            document_splitter_fn=document_splitter_fn,
            preprocessing_fn=preprocessing_fns,
        )
        return cls(
            colbert_model=colbert_model_instance,
            corpus_processor=corpus_processor_instance,
            verbose=verbose,
        )

    @classmethod
    def from_index(
        cls,
        index_path: Union[str, Path],
        n_gpu: int = -1,
        verbose: int = 1,
        document_splitter_fn: Optional[Callable] = llama_index_sentence_splitter,
        preprocessing_fns: Optional[Union[Callable, List[Callable]]] = None,
        pretrained_model_name_or_path: Optional[Union[str, Path]] = None,
        **colbert_kwargs: Any,
    ):
        """
        Load a RAGPretrainedModel from an existing index directory.
        The underlying ColBERT model will be loaded, and this specific index will be
        set as the initially loaded index. The model will have access to all other
        indexes in the same experiment.

        Parameters:
            index_path: Path to the specific ColBERT index directory.
            n_gpu: Number of GPUs to use.
            verbose: Verbosity level.
            document_splitter_fn: Function to split documents into passages.
            preprocessing_fns: Functions to preprocess passages.
            pretrained_model_name_or_path: Optional. Path/name of the base ColBERT model.
                                           If not provided, attempts to infer from index metadata.
            **colbert_kwargs: Additional keyword arguments for ColBERTConfig.
        """
        resolved_index_path = Path(index_path).resolve()
        initial_index_name = resolved_index_path.name

        # Always infer index_root and experiment_name from the path structure
        # Expected structure: index_root/experiment_name/indexes/index_name
        if len(resolved_index_path.parts) < 3 or resolved_index_path.parent.name != "indexes":
            raise ValueError(f"Invalid index path structure: {resolved_index_path}. Expected: .../experiment_name/indexes/index_name")

        experiment_name_for_colbert = resolved_index_path.parent.parent.name
        index_root_for_colbert = str(resolved_index_path.parent.parent.parent)

        base_model_path_to_use = pretrained_model_name_or_path

        if not base_model_path_to_use:
            try:
                temp_colbert_config = ColBERTConfig.load_from_index(str(resolved_index_path))
                base_model_path_to_use = temp_colbert_config.checkpoint
                if verbose > 0:
                    print(f"Inferred base model: {base_model_path_to_use} from index '{initial_index_name}'.")
            except Exception as e:
                if verbose > 0:
                    print(
                        f"Warning: Could not automatically determine base model path "
                        f"from index '{index_path}'. Error: {e}"
                    )
                raise ValueError(
                    "Failed to load from index. pretrained_model_name_or_path is required "
                    "if it cannot be inferred from the index's metadata (e.g., ColBERTConfig in the index)."
                ) from e

        if not base_model_path_to_use:
            raise ValueError("pretrained_model_name_or_path could not be determined for from_index.")

        if verbose > 0:
            print(
                f"Loading RAG model from index '{initial_index_name}' with "
                f"experiment: {experiment_name_for_colbert}, index_root: {index_root_for_colbert}"
            )

        # Allow kwargs to override inferred values if explicitly passed
        if "index_root" in colbert_kwargs:
            index_root_for_colbert = colbert_kwargs.pop("index_root")
        if "experiment_name" in colbert_kwargs:
             experiment_name_for_colbert = colbert_kwargs.pop("experiment_name")

        return cls.from_pretrained(
            pretrained_model_name_or_path=str(base_model_path_to_use),
            index_root=index_root_for_colbert,
            initial_index_name=initial_index_name,
            n_gpu=n_gpu,
            verbose=verbose,
            experiment_name=experiment_name_for_colbert,
            document_splitter_fn=document_splitter_fn,
            preprocessing_fns=preprocessing_fns,
            **colbert_kwargs,
        )

    def _build_docid_metadata_map(
        self,
        document_ids: List[str], # Assumes these are the original document IDs
        document_metadatas: Optional[List[Optional[dict]]],
    ) -> Optional[Dict[str, Any]]:
        if document_metadatas is None or not any(document_metadatas):
            return None

        if not document_ids:
            if self.verbose > 0:
                print("Warning: _build_docid_metadata_map called with empty document_ids.")
            return None

        docid_metadata_map: Dict[str, Any] = {}
        for i, doc_id in enumerate(document_ids):
            if i < len(document_metadatas) and document_metadatas[i] is not None:
                docid_metadata_map[doc_id] = document_metadatas[i]

        return docid_metadata_map if docid_metadata_map else None

    def _prepare_documents_for_processing(
        self,
        documents: List[str],
        document_ids: Optional[List[str]] = None,
        document_metadatas: Optional[List[Optional[dict]]] = None,
        **splitter_kwargs: Any,
    ) -> Tuple[List[str], Dict[int, str], Optional[Dict[str, Any]]]:
        """
        Core document processing: splits documents, creates passage collection,
        pid_docid_map, and docid_metadata_map.
        """
        final_original_doc_ids = document_ids if document_ids is not None else [str(uuid4()) for _ in documents]
        if len(final_original_doc_ids) != len(documents):
            raise ValueError("Mismatch between number of documents and provided document_ids.")
        if len(set(final_original_doc_ids)) != len(final_original_doc_ids):
            raise ValueError("Provided document_ids must be unique.")


        processed_chunks_dicts: List[Dict] = self.corpus_processor.process_corpus(
            documents=documents,
            document_ids=final_original_doc_ids,
            **splitter_kwargs,
        )

        if not processed_chunks_dicts:
            return [], {}, None

        collection_for_colbert: List[str] = [chunk['content'] for chunk in processed_chunks_dicts]
        pid_docid_map_for_colbert: Dict[int, str] = {
            i: chunk['document_id'] for i, chunk in enumerate(processed_chunks_dicts)
        }

        docid_metadata_map_for_colbert = self._build_docid_metadata_map(
            final_original_doc_ids, document_metadatas
        )

        return collection_for_colbert, pid_docid_map_for_colbert, docid_metadata_map_for_colbert

    def index(
        self,
        index_name: str,
        documents: List[str],
        document_ids: Optional[List[str]] = None,
        document_metadatas: Optional[List[Optional[dict]]] = None,
        max_document_length: int = 256, # Used as chunk_size by default splitter
        overwrite_index: Union[bool, str] = "reuse",
        bsize: int = 32,
        use_faiss_index: bool = False,
        **colbert_index_kwargs: Any # For additional ColBERTConfig settings for this index
    ) -> str:
        """
        Build an index from a list of documents for the specified index_name.

        Parameters:
            index_name: The name of the index to build or update.
            documents: The list of documents (full texts) to index.
            document_ids: Optional list of unique IDs for the original documents.
            document_metadatas: Optional list of metadata dicts for original documents.
            max_document_length: Target chunk size for splitting.
            overwrite_index: Policy for overwriting if index exists ('reuse', True, False, 'force_silent_overwrite').
            bsize: Batch size for ColBERT encoding.
            use_faiss_index: Whether to use FAISS for the index backend (if applicable).
            **colbert_index_kwargs: Additional kwargs for ColBERT's index method (e.g., nbits).

        Returns:
            The path to the created or updated index.
        """
        if not documents:
            if self.verbose > 0:
                print(f"No documents provided for indexing '{index_name}'. Skipping.")
            return self.model._get_index_path(index_name)

        collection, pid_docid_map, docid_metadata_map = self._prepare_documents_for_processing(
            documents, document_ids, document_metadatas, chunk_size=max_document_length
        )

        if not collection:
            if self.verbose > 0:
                print(f"Document processing resulted in an empty collection for index '{index_name}'. Skipping ColBERT indexing.")
            return self.model._get_index_path(index_name)

        return self.model.index(
            index_name=index_name,
            collection=collection,
            pid_docid_map=pid_docid_map,
            docid_metadata_map=docid_metadata_map,
            max_document_length=max_document_length, # This influences ColBERTConfig.doc_maxlen for the index
            overwrite=overwrite_index,
            bsize=bsize,
            use_faiss=use_faiss_index,
            **colbert_index_kwargs
        )

    def add_to_index(
        self,
        index_name: str,
        new_documents: List[str],
        new_document_ids: Optional[List[str]] = None,
        new_document_metadatas: Optional[List[Optional[dict]]] = None,
        bsize: int = 32,
        max_document_length: int = 256, # Target chunk size for splitting new documents
    ):
        """
        Add new documents to an existing index.

        Parameters:
            index_name: The name of the index to add to.
            new_documents: List of new documents (full texts) to add.
            new_document_ids: Optional list of unique IDs for the new original documents.
            new_document_metadatas: Optional list of metadata for new original documents.
            bsize: Batch size for ColBERT encoding.
            max_document_length: Target chunk size for splitting new documents.
        """
        if not new_documents:
            if self.verbose > 0:
                print(f"No new documents provided to add to index '{index_name}'. Skipping.")
            return

        # chunk_size for splitter from max_document_length
        passages_for_colbert, pid_to_docid_for_new_batch, docid_to_metadata_for_new_batch = \
            self._prepare_documents_for_processing(
                new_documents, new_document_ids, new_document_metadatas, chunk_size=max_document_length
            )

        if not passages_for_colbert:
            if self.verbose > 0:
                print(f"Processing new documents for index '{index_name}' resulted in no passages. Nothing to add.")
            return

        self.model.add_to_index(
            index_name=index_name,
            new_documents=passages_for_colbert,
            new_pid_docid_map_for_new_docs=pid_to_docid_for_new_batch,
            new_docid_metadata_map=docid_to_metadata_for_new_batch,
            bsize=bsize,
        )

    def delete_from_index(
        self,
        index_name: str,
        document_ids: Union[str, List[str]],
    ):
        """
        Delete documents (and all their associated passages) from a specified index.

        Parameters:
            index_name: The name of the index from which to delete.
            document_ids: A single document ID or a list of document IDs to delete.
        """
        doc_ids_list = [document_ids] if isinstance(document_ids, str) else document_ids
        if not doc_ids_list:
            if self.verbose > 0:
                print(f"No document IDs provided for deletion from index '{index_name}'. Skipping.")
            return
        self.model.delete_from_index(index_name=index_name, document_ids=doc_ids_list)

    def search(
        self,
        index_name: str,
        query: Union[str, List[str]],
        k: int = 10,
        force_fast: bool = False,
        zero_index_ranks: bool = False,
        doc_ids_filter: Optional[List[str]] = None,
    ) -> Union[List[dict], List[List[dict]]]:
        """
        Search an index for a given query or list of queries.

        Parameters:
            index_name: The name of the index to search.
            query: The query string or list of query strings.
            k: The number of results to return per query.
            force_fast: Use faster, potentially less accurate search settings.
            zero_index_ranks: If True, ranks are 0-indexed. Otherwise 1-indexed.
            doc_ids_filter: Optional list of document IDs to restrict the search to.

        Returns:
            A list of result dictionaries for a single query, or a list of lists of
            result dictionaries for multiple queries.
        """
        return self.model.search(
            index_name=index_name,
            query=query,
            k=k,
            force_fast=force_fast,
            zero_index_ranks=zero_index_ranks,
            doc_ids=doc_ids_filter,
        )

    def rerank(
        self,
        query: str,
        documents: List[str],
        k: int = 10,
        zero_index_ranks: bool = False,
        bsize: Union[Literal["auto"], int] = "auto",
        max_tokens: Union[Literal["auto"], int] = "auto",
    ) -> List[dict]:
        """
        Rerank a list of documents in-memory for a given query.

        Parameters:
            query: The query string.
            documents: A list of document contents (passages) to rerank.
            k: The number of top documents to return.
            zero_index_ranks: If True, ranks are 0-indexed.
            bsize: Batch size for encoding.
            max_tokens: Max tokens for document encoding during this operation.

        Returns:
            A list of reranked result dictionaries.
        """
        return self.model.rank(
            query=query,
            documents=documents,
            k=k,
            zero_index_ranks=zero_index_ranks,
            bsize=bsize,
            max_tokens=max_tokens,
        )

    def encode(
        self,
        documents: List[str],
        document_metadatas: Optional[List[Optional[dict]]] = None,
        bsize: int = 32,
        max_tokens: Union[Literal["auto"], int] = "auto",
        verbose_override: Optional[bool] = None,
    ):
        """
        Encode documents and cache their embeddings in memory for later
        `search_encoded_docs` calls.

        Parameters:
            documents: List of document contents to encode.
            document_metadatas: Optional list of metadata dicts, matched to documents.
            bsize: Batch size for encoding.
            max_tokens: Max tokens for document encoding. 'auto' adjusts based on content.
            verbose_override: Override instance verbosity for this call.
        """
        self.model.encode(
            documents=documents,
            document_metadatas=document_metadatas,
            bsize=bsize,
            max_tokens=max_tokens,
            verbose_override=verbose_override,
        )

    def search_encoded_docs(
        self,
        queries: Union[str, List[str]],
        k: int = 10,
        bsize: int = 32,
        zero_index_ranks: bool = False,
    ) -> Union[List[dict], List[List[dict]]]:
        """
        Search through documents previously encoded and cached in memory via `encode()`.

        Parameters:
            queries: The query string or list of query strings.
            k: The number of results to return per query.
            bsize: Batch size for query encoding.
            zero_index_ranks: If True, ranks are 0-indexed.

        Returns:
            Search results, similar to the `search` method.
        """
        return self.model.search_encoded_docs(
            queries=queries, k=k, bsize=bsize, zero_index_ranks=zero_index_ranks
        )

    def clear_encoded_docs(self, force: bool = False):
        """
        Clear any documents and their embeddings cached in memory by `encode()`.

        Parameters:
            force: If True, clears without a confirmation delay.
        """
        self.model.clear_encoded_docs(force=force)

    def get_model(self) -> ColBERT:
        """Returns the underlying ColBERT model instance."""
        return self.model

    def get_available_indexes(self) -> Dict[str, Dict[str, Union[int, bool, None]]]:
        """Returns a list of available indexes."""
        return self.model.list_available_indexes()
