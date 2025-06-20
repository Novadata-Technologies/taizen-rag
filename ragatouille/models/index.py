import time
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Literal, Optional, TypeVar, Union

import srsly
import torch
from colbert import Indexer, IndexUpdater, Searcher
from colbert.indexing.collection_indexer import CollectionIndexer
from colbert.infra import ColBERTConfig

from ragatouille.models import torch_kmeans

IndexType = Literal["FLAT", "HNSW", "PLAID"]


class ModelIndex(ABC):
    index_type: IndexType

    def __init__(
        self,
        config: ColBERTConfig,
    ) -> None:
        self.config = config

    @staticmethod
    @abstractmethod
    def construct(
        config: ColBERTConfig,
        checkpoint: str,
        collection: List[str],
        index_name: Optional["str"] = None,
        overwrite: Union[bool, str] = "reuse",
        verbose: bool = True,
        **kwargs,
    ) -> "ModelIndex": ...

    @staticmethod
    @abstractmethod
    def load_from_file(
        index_path: str,
        index_name: Optional[str],
        index_config: dict[str, Any],
        config: ColBERTConfig,
        verbose: bool = True,
    ) -> "ModelIndex": ...

    @abstractmethod
    def build(
        self,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_name: Optional["str"] = None,
        overwrite: Union[bool, str] = "reuse",
        verbose: bool = True,
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        config: ColBERTConfig,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_name: Optional[str],
        base_model_max_tokens: int,
        query: Union[str, list[str]],
        k: int = 10,
        pids: Optional[List[int]] = None,
        force_reload: bool = False,
        **kwargs,
    ) -> list[tuple[list, list, list]]: ...

    @abstractmethod
    def _search(self, query: str, k: int, pids: Optional[List[int]] = None): ...

    @abstractmethod
    def _batch_search(self, query: list[str], k: int): ...

    @abstractmethod
    def add(
        self,
        config: ColBERTConfig,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_root: str,
        index_name: str,
        new_collection: List[str],
        verbose: bool = True,
        **kwargs,
    ) -> None: ...

    @abstractmethod
    def delete(
        self,
        config: ColBERTConfig,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_name: str,
        pids_to_remove: Union[TypeVar("T"), List[TypeVar("T")]],
        verbose: bool = True,
    ) -> None: ...

    @abstractmethod
    def _export_config(self) -> dict[str, Any]: ...

    def export_metadata(self) -> dict[str, Any]:
        config = self._export_config()
        config["index_type"] = self.index_type
        return config


class FLATModelIndex(ModelIndex):
    index_type = "FLAT"


class HNSWModelIndex(ModelIndex):
    index_type = "HNSW"


class PLAIDModelIndex(ModelIndex):
    _DEFAULT_INDEX_BSIZE = 32
    index_type = "PLAID"
    faiss_kmeans = staticmethod(deepcopy(CollectionIndexer._train_kmeans))
    pytorch_kmeans = staticmethod(torch_kmeans._train_kmeans)

    def __init__(self, config: ColBERTConfig) -> None:
        super().__init__(config)
        self.searcher: Optional[Searcher] = None

    @staticmethod
    def construct(
        config: ColBERTConfig,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_name: Optional["str"] = None,
        overwrite: Union[bool, str] = "reuse",
        verbose: bool = True,
        **kwargs,
    ) -> "PLAIDModelIndex":
        return PLAIDModelIndex(config).build(
            checkpoint, collection, index_name, overwrite, verbose, **kwargs
        )

    @staticmethod
    def load_from_file(
        index_path: str,
        index_name: Optional[str],
        index_config: dict[str, Any],
        config: ColBERTConfig,
        verbose: bool = True,
    ) -> "PLAIDModelIndex":
        _, _, _, _ = index_path, index_name, index_config, verbose
        return PLAIDModelIndex(config)

    def build(
        self,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_name: Optional["str"] = None,
        overwrite: Union[bool, str] = "reuse",
        verbose: bool = True,
        **kwargs,
    ) -> "PLAIDModelIndex":
        bsize = kwargs.get("bsize", PLAIDModelIndex._DEFAULT_INDEX_BSIZE)
        assert isinstance(bsize, int)

        nbits = 2
        if len(collection) < 10000:
            nbits = 4
        self.config = ColBERTConfig.from_existing(
            self.config, ColBERTConfig(nbits=nbits, index_bsize=bsize)
        )

        # Instruct colbert-ai to disable forking if nranks == 1
        self.config.avoid_fork_if_possible = True

        if len(collection) > 100000:
            self.config.kmeans_niters = 4
        elif len(collection) > 50000:
            self.config.kmeans_niters = 10
        else:
            self.config.kmeans_niters = 20

        # Monkey-patch colbert-ai to avoid using FAISS
        monkey_patching = (
            len(collection) < 75000 and kwargs.get("use_faiss", False) is False
        )
        if monkey_patching:
            print(
                "---- WARNING! You are using PLAID with an experimental replacement for FAISS for greater compatibility ----"
            )
            print("This is a behaviour change from RAGatouille 0.8.0 onwards.")
            print(
                "This works fine for most users and smallish datasets, but can be considerably slower than FAISS and could cause worse results in some situations."
            )
            print(
                "If you're confident with FAISS working on your machine, pass use_faiss=True to revert to the FAISS-using behaviour."
            )
            print("--------------------")
            CollectionIndexer._train_kmeans = self.pytorch_kmeans

            # Try to keep runtime stable -- these are values that empirically didn't degrade performance at all on 3 benchmarks.
            # More tests required before warning can be removed.
            try:
                indexer = Indexer(
                    checkpoint=checkpoint,
                    config=self.config,
                    verbose=verbose,
                )
                indexer.configure(avoid_fork_if_possible=True)

                indexer.index(
                    name=index_name, collection=collection, overwrite=overwrite
                )
            except Exception as err:
                print(
                    f"PyTorch-based indexing did not succeed with error: {err}",
                    "! Reverting to using FAISS and attempting again...",
                )
                monkey_patching = False
        if monkey_patching is False:
            CollectionIndexer._train_kmeans = self.faiss_kmeans
            if torch.cuda.is_available():
                import faiss

                if not hasattr(faiss, "StandardGpuResources"):
                    print(
                        "________________________________________________________________________________\n"
                        "WARNING! You have a GPU available, but only `faiss-cpu` is currently installed.\n",
                        "This means that indexing will be slow. To make use of your GPU.\n"
                        "Please install `faiss-gpu` by running:\n"
                        "pip uninstall --y faiss-cpu & pip install faiss-gpu\n",
                        "________________________________________________________________________________",
                    )
                    print("Will continue with CPU indexing in 5 seconds...")
                    time.sleep(5)
            indexer = Indexer(
                checkpoint=checkpoint,
                config=self.config,
                verbose=verbose,
            )
            indexer.configure(avoid_fork_if_possible=True)
            indexer.index(name=index_name, collection=collection, overwrite=overwrite)

        return self

    def _load_searcher(
        self,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_name: Optional[str],
        force_fast: bool = False,
    ):
        print(
            f"Loading searcher for index {index_name} for the first time...",
            "This may take a few seconds",
        )
        self.searcher = Searcher(
            checkpoint=checkpoint,
            config=None,
            collection=collection,
            index_root=self.config.root,
            index=index_name,
        )

        if not force_fast:
            self.searcher.configure(ndocs=1024)
            self.searcher.configure(ncells=16)
            if len(self.searcher.collection) < 10000:
                self.searcher.configure(ncells=8)
                self.searcher.configure(centroid_score_threshold=0.4)
            elif len(self.searcher.collection) < 100000:
                self.searcher.configure(ncells=4)
                self.searcher.configure(centroid_score_threshold=0.45)
            # Otherwise, use defaults for k
        else:
            # Use fast settingss
            self.searcher.configure(ncells=1)
            self.searcher.configure(centroid_score_threshold=0.5)
            self.searcher.configure(ndocs=256)

        print("Searcher loaded!")

    def _search(self, query: str, k: int, pids: Optional[List[int]] = None):
        assert self.searcher is not None
        return self.searcher.search(query, k=k, pids=pids)

    def _batch_search(self, query: list[str], k: int):
        assert self.searcher is not None
        queries = {i: x for i, x in enumerate(query)}
        results = self.searcher.search_all(queries, k=k)
        results = [
            [list(zip(*value))[i] for i in range(3)]
            for value in results.todict().values()
        ]
        return results

    def _upgrade_searcher_maxlen(self, maxlen: int, base_model_max_tokens: int):
        assert self.searcher is not None
        # Keep maxlen stable at 32 for short queries for easier visualisation
        maxlen = min(max(maxlen, 32), base_model_max_tokens)
        self.searcher.config.query_maxlen = maxlen
        self.searcher.checkpoint.query_tokenizer.query_maxlen = maxlen

    def search(
        self,
        config: ColBERTConfig,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_name: Optional[str],
        base_model_max_tokens: int,
        query: Union[str, list[str]],
        k: int = 10,
        pids: Optional[List[int]] = None,
        force_reload: bool = False,
        **kwargs,
    ) -> list[tuple[list, list, list]]:
        self.config = config

        force_fast = kwargs.get("force_fast", False)
        assert isinstance(force_fast, bool)

        if self.searcher is None or force_reload:
            self._load_searcher(
                checkpoint,
                collection,
                index_name,
                force_fast,
            )
        assert self.searcher is not None

        base_ncells = self.searcher.config.ncells
        base_ndocs = self.searcher.config.ndocs

        if k > len(self.searcher.collection):
            print(
                "WARNING: k value is larger than the number of documents in the index!",
                f"Lowering k to {len(self.searcher.collection)}...",
            )
            k = len(self.searcher.collection)

        # For smaller collections, we need a higher ncells value to ensure we return enough results
        if k > (32 * self.searcher.config.ncells):
            self.searcher.configure(ncells=min((k // 32 + 2), base_ncells))

        self.searcher.configure(ndocs=max(k * 4, base_ndocs))

        if isinstance(query, str):
            query_length = int(len(query.split(" ")) * 1.35)
            self._upgrade_searcher_maxlen(query_length, base_model_max_tokens)
            results = [self._search(query, k, pids)]
        else:
            longest_query_length = max([int(len(x.split(" ")) * 1.35) for x in query])
            self._upgrade_searcher_maxlen(longest_query_length, base_model_max_tokens)
            results = self._batch_search(query, k)

        # Restore original ncells&ndocs if it had to be changed for large k values
        self.searcher.configure(ncells=base_ncells)
        self.searcher.configure(ndocs=base_ndocs)

        return results  # type: ignore

    @staticmethod
    def _should_rebuild(current_len: int, new_doc_len: int) -> bool:
        """
        Heuristic to determine if it is more efficient to rebuild the index instead of updating it.
        """
        return current_len + new_doc_len < 5000 or new_doc_len > current_len * 0.05

    def add(
        self,
        config: ColBERTConfig,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_root: str,
        index_name: str,
        new_collection: List[str],
        verbose: bool = True,
        **kwargs,
    ) -> None:
        self.config = config

        bsize = kwargs.get("bsize", PLAIDModelIndex._DEFAULT_INDEX_BSIZE)
        assert isinstance(bsize, int)

        #searcher = Searcher(
        #    checkpoint=checkpoint,
        #    config=None,
        #    collection=collection, # This `collection` is the full final collection (old + new)
        #    index=index_name,
        #    index_root=index_root, # This should be consistent with self.config.root
        #    verbose=False, # Less verbose for this temporary searcher
        #)

        old_collection_len = len(collection) - len(new_collection)

        if PLAIDModelIndex._should_rebuild(old_collection_len, len(new_collection)):
            if verbose:
                print(f"Rebuilding index {index_name} due to new additions. Old size: {old_collection_len}, New passages: {len(new_collection)}")
            # build needs the full, final collection
            self.build(
                checkpoint=checkpoint,
                collection=collection, # This is already the final combined collection
                index_name=index_name,
                overwrite="force_silent_overwrite",
                verbose=verbose,
                **kwargs, # Pass bsize (already in self.config.index_bsize if updated), use_faiss etc. to build
            )
        else:
            if verbose:
                print(f"Updating existing index {index_name} with {len(new_collection)} new passages.")
            # IndexUpdater needs a searcher that reflects the state of the index *before* adding new_collection.
            # This searcher is used for its configuration and to point to the existing index parts.
            collection_before_add = collection[:-len(new_collection)] if old_collection_len > 0 else []

            # Initialize a searcher context for the IndexUpdater
            # This searcher should point to the index *as it exists on disk before this update*
            searcher_for_updater_config = ColBERTConfig.from_existing(self.config, ColBERTConfig())
            searcher_for_updater_config.root = index_root # ensure it points to the correct place.
            searcher_for_updater_config.index_name = index_name

            searcher_for_updater = Searcher(
                checkpoint=checkpoint,
                config=searcher_for_updater_config,
                collection=collection_before_add, # Collection before adding new docs
                index=index_name, # Redundant if in config, but safe
                # index_root is implicitly handled by searcher_for_updater_config.root
                verbose=False,
            )
            updater = IndexUpdater(
                config=self.config, # This config has potentially updated bsize
                searcher=searcher_for_updater,
                checkpoint=checkpoint
            )
            updater.add(new_collection) # updater.add processes only the new docs
            updater.persist_to_disk()

        self.searcher = None # Invalidate the main searcher for PLAIDModelIndex
        if verbose:
            print(f"Finished add operation for index {index_name}. Searcher invalidated, will reload on next search.")

    def delete(
        self,
        config: ColBERTConfig,
        checkpoint: Union[str, Path],
        collection: List[str],
        index_name: str,
        pids_to_remove: Union[TypeVar("T"), List[TypeVar("T")]],
        verbose: bool = True,
    ) -> None:
        self.config = config

        # Initialize the searcher and updater
        searcher = Searcher(
            checkpoint=checkpoint,
            config=None,
            collection=collection, # The collection as it is *before* these pids are removed
            index=index_name,
            index_root=self.config.root, # ensure consistency with self.config
            verbose=verbose, # can be verbose
        )
        updater = IndexUpdater(config=config, searcher=searcher, checkpoint=checkpoint)

        updater.remove(pids_to_remove)
        updater.persist_to_disk()

        self.searcher = None # Invalidate the main searcher for PLAIDModelIndex
        if verbose:
            print(f"Finished delete operation for index {index_name}. Searcher invalidated, will reload on next search.")

    def _export_config(self) -> dict[str, Any]:
        return {}


class ModelIndexFactory:
    _MODEL_INDEX_BY_NAME = {
        "FLAT": FLATModelIndex,
        "HNSW": HNSWModelIndex,
        "PLAID": PLAIDModelIndex,
    }

    @staticmethod
    def _raise_if_invalid_index_type(index_type: str) -> IndexType:
        if index_type not in ["FLAT", "HNSW", "PLAID"]:
            raise ValueError(
                f"Unsupported index_type `{index_type}`; it must be one of 'FLAT', 'HNSW', OR 'PLAID'"
            )
        return index_type  # type: ignore

    @staticmethod
    def construct(
        index_type: Union[Literal["auto"], IndexType],
        config: ColBERTConfig,
        checkpoint: str,
        collection: List[str],
        index_name: Optional["str"] = None,
        overwrite: Union[bool, str] = "reuse",
        verbose: bool = True,
        **kwargs,
    ) -> ModelIndex:
        # Automatically choose the appropriate index for the desired "workload".
        if index_type == "auto":
            # NOTE: For now only PLAID indexes are supported.
            index_type = "PLAID"
        return ModelIndexFactory._MODEL_INDEX_BY_NAME[
            ModelIndexFactory._raise_if_invalid_index_type(index_type)
        ].construct(
            config, checkpoint, collection, index_name, overwrite, verbose, **kwargs
        )

    @staticmethod
    def load_from_file(
        index_path: str,
        index_name: Optional[str],
        config: ColBERTConfig,
        verbose: bool = True,
    ) -> ModelIndex:
        metadata = srsly.read_json(index_path + "/metadata.json")
        try:
            index_config = metadata["RAGatouille"]["index_config"]  # type: ignore
        except KeyError:
            if verbose:
                print(
                    f"Constructing default index configuration for index `{index_name}` as it does not contain RAGatouille specific metadata."
                )
            index_config = {
                "index_type": "PLAID",
                "index_name": index_name,
            }
        index_name = (
            index_name if index_name is not None else index_config["index_name"]  # type: ignore
        )
        return ModelIndexFactory._MODEL_INDEX_BY_NAME[
            ModelIndexFactory._raise_if_invalid_index_type(index_config["index_type"])  # type: ignore
        ].load_from_file(index_path, index_name, index_config, config, verbose)
