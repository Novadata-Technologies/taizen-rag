from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Literal


class LateInteractionModel(ABC):
    @abstractmethod
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        index_root: Optional[str] = None,
        n_gpu: int = -1,
        verbose: int = 1,
        initial_index_name: Optional[str] = None,
        experiment_name: str = "colbert",
        **kwargs: Any,
    ):
        """
        Initializes the late-interaction model.

        Parameters:
            pretrained_model_name_or_path: Path to a local checkpoint or Hugging Face model name.
            index_root: The root directory where all named indices will be stored/loaded from.
            n_gpu: Number of GPUs to use. -1 for all available.
            verbose: Verbosity level.
            initial_index_name: Optionally, the name of an index to load immediately.
            experiment_name: Name of the experiment, used in structuring the index path.
            **kwargs: Additional keyword arguments for model configuration.
        """
        ...

    @abstractmethod
    def index(
        self,
        index_name: str,
        collection: List[str],
        pid_docid_map: Dict[int, str],
        docid_metadata_map: Optional[Dict[str, Any]] = None,
        max_document_length: int = 256,
        overwrite: Union[bool, str] = "reuse",
        bsize: int = 32,
        use_faiss: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Builds or updates an index for the given collection of passages.

        Parameters:
            index_name: The name of the index to build or update.
            collection: List of passages/chunks to index.
            pid_docid_map: Mapping from passage ID (0-indexed within collection) to original document ID.
            docid_metadata_map: Optional mapping from original document ID to its metadata.
            max_document_length: Maximum document length for configuring the index (e.g., doc_maxlen).
            overwrite: Policy for overwriting if index exists.
            bsize: Batch size for encoding passages.
            use_faiss: Whether to use FAISS for the index backend (if applicable).
            **kwargs: Additional keyword arguments for index-specific configuration.

        Returns:
            The path to the created or updated index.
        """
        ...

    @abstractmethod
    def add_to_index(
        self,
        index_name: str,
        new_documents: List[str],
        new_pid_docid_map_for_new_docs: Dict[int, str],
        new_docid_metadata_map: Optional[Dict[str, Any]] = None,
        bsize: int = 32,
        use_faiss: bool = False,
    ) -> None:
        """
        Adds new passages to an existing index.

        Parameters:
            index_name: The name of the index to add to.
            new_documents: List of new passages/chunks to add.
            new_pid_docid_map_for_new_docs: Mapping for the new passages (0-indexed for new_documents)
                                             to their original document IDs.
            new_docid_metadata_map: Optional metadata for original document IDs corresponding to new_documents.
            bsize: Batch size for encoding new passages.
            use_faiss: Whether to use FAISS if the add operation triggers a rebuild.
        """
        ...

    @abstractmethod
    def delete_from_index(
        self,
        index_name: str,
        document_ids: Union[str, List[str]],
    ) -> None:
        """
        Deletes documents (and all their associated passages) from a specified index.

        Parameters:
            index_name: The name of the index from which to delete.
            document_ids: A single original document ID or a list of original document IDs to delete.
        """
        ...

    @abstractmethod
    def search(
        self,
        index_name: str,
        query: Union[str, List[str]],
        k: int = 10,
        force_fast: bool = False,
        zero_index_ranks: bool = False,
        doc_ids: Optional[List[str]] = None,
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Searches an index for a given query or list of queries.

        Parameters:
            index_name: The name of the index to search.
            query: The query string or list of query strings.
            k: The number of results to return per query.
            force_fast: Use faster, potentially less accurate search settings.
            zero_index_ranks: If True, ranks are 0-indexed. Otherwise 1-indexed.
            doc_ids: Optional list of original document IDs to restrict the search to.

        Returns:
            A list of result dictionaries for a single query, or a list of lists of
            result dictionaries for multiple queries.
        """
        ...

    @abstractmethod
    def train(
        self,
        index_name_for_config: Optional[str],
        data_dir: Union[str, Path],
        training_config_overrides: Dict[str, Any],
    ) -> str:
        """
        Trains the model.

        Parameters:
            index_name_for_config: Optional name of an existing index to use its configuration as a base.
            data_dir: Path to the directory containing training data (triples, queries, corpus).
            training_config_overrides: Dictionary of parameters to override in the model's training configuration.

        Returns:
            Path to the best trained checkpoint.
        """
        ...

    @abstractmethod
    def rank(
        self,
        query: str,
        documents: List[str],
        k: int = 10,
        zero_index_ranks: bool = False,
        bsize: Union[Literal["auto"], int] = "auto",
        max_tokens: Union[Literal["auto"], int] = "auto",
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of documents in-memory for a given query without using a disk-based index.

        Parameters:
            query: The query string.
            documents: A list of document contents (passages) to rerank.
            k: The number of top documents to return.
            zero_index_ranks: If True, ranks are 0-indexed.
            bsize: Batch size for encoding.
            max_tokens: Max tokens for document encoding. 'auto' adjusts based on content.

        Returns:
            A list of reranked result dictionaries.
        """
        ...

    @abstractmethod
    def encode(
        self,
        documents: List[str],
        document_metadatas: Optional[List[Optional[Dict[str, Any]]]] = None,
        bsize: int = 32,
        max_tokens: Union[Literal["auto"], int] = "auto",
        verbose_override: Optional[bool] = None,
    ) -> None:
        """
        Encodes documents and caches their embeddings in memory.

        Parameters:
            documents: List of document contents to encode.
            document_metadatas: Optional list of metadata dicts, matched to documents.
            bsize: Batch size for encoding.
            max_tokens: Max tokens for document encoding. 'auto' adjusts based on content.
            verbose_override: Override instance verbosity for this call.
        """
        ...

    @abstractmethod
    def search_encoded_docs(
        self,
        queries: Union[str, List[str]],
        k: int = 10,
        bsize: int = 32,
        zero_index_ranks: bool = False,
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Searches through documents previously encoded and cached in memory via `encode()`.

        Parameters:
            queries: The query string or list of query strings.
            k: The number of results to return per query.
            bsize: Batch size for query encoding.
            zero_index_ranks: If True, ranks are 0-indexed.

        Returns:
            Search results, similar to the `search` method for indexed data.
        """
        ...

    @abstractmethod
    def clear_encoded_docs(self, force: bool = False) -> None:
        """
        Clears any documents and their embeddings cached in memory by `encode()`.

        Parameters:
            force: If True, clears without a confirmation delay.
        """
        ...