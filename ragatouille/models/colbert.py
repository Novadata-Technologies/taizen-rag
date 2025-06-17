import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, TypeVar, Union, Any

import numpy as np
import srsly
import torch
from colbert import Trainer
from colbert.infra import ColBERTConfig, Run, RunConfig
from colbert.modeling.checkpoint import Checkpoint

from ragatouille.models.base import LateInteractionModel
from ragatouille.models.index import ModelIndex, ModelIndexFactory

# TODO: Move all bsize related calcs to `_set_bsize()`


class ColBERT(LateInteractionModel):
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        index_root: Optional[str] = None,
        n_gpu: int = -1,
        verbose: int = 1,
        initial_index_name: Optional[str] = None,
        experiment_name: str = "colbert", # Default experiment name
        **kwargs, # Catch-all for other ColBERTConfig settings for new indices
    ):
        self.verbose = verbose
        self.base_pretrained_model_name_or_path = pretrained_model_name_or_path
        self.index_root = index_root if index_root is not None else ".ragatouille/"
        self.experiment_name = experiment_name

        self.base_model_max_tokens = 510 # Default, will be updated after model load

        if n_gpu == -1:
            n_gpu = 1 if torch.cuda.device_count() == 0 else torch.cuda.device_count()

        # Multi-index storage
        self.collections: Dict[str, List[str]] = {}
        self.pid_docid_maps: Dict[str, Dict[int, str]] = {}
        self.docid_pid_maps: Dict[str, Dict[str, List[int]]] = {}
        self.docid_metadata_maps: Dict[str, Optional[Dict[str, Any]]] = {}
        self.model_indices: Dict[str, Optional[ModelIndex]] = {}
        self.index_configs: Dict[str, ColBERTConfig] = {} # Stores specific config for each index
        self.index_paths: Dict[str, str] = {} # Stores full path to each index directory

        # Load base model configuration
        # For new indices, this base_config will be the starting point
        self.base_model_config = ColBERTConfig.load_from_checkpoint(
            str(pretrained_model_name_or_path),
            **kwargs # Pass other ColBERTConfig settings
        )
        self.base_model_config.root = str(Path(self.index_root) / self.experiment_name / "indexes")
        self.base_model_config.experiment = self.experiment_name


        # Global RunConfig for the underlying ColBERT model
        self.run_config = RunConfig(
            nranks=n_gpu, experiment=self.experiment_name, root=self.index_root
        )

        # Load the actual inference model weights (once)
        self.inference_ckpt = Checkpoint(
            str(self.base_pretrained_model_name_or_path), colbert_config=self.base_model_config
        )
        self.base_model_max_tokens = (
            self.inference_ckpt.bert.config.max_position_embeddings
        ) - 4 # Standard adjustment

        self.run_context = Run().context(self.run_config)
        self.run_context.__enter__()  # Manually enter the context

        if initial_index_name:
            if not self.index_root:
                raise ValueError("index_root must be provided if initial_index_name is set.")
            try:
                self._get_or_load_index_context(initial_index_name, create_if_not_exists=False)
                if self.verbose > 0:
                    print(f"Successfully loaded initial index '{initial_index_name}'.")
            except FileNotFoundError:
                if self.verbose > 0:
                    print(f"Initial index '{initial_index_name}' not found at expected location. It will need to be created via .index() method.")
            except Exception as e:
                if self.verbose > 0:
                    print(f"Failed to load initial index '{initial_index_name}': {e}")


    def _get_index_path(self, index_name: str) -> str:
        return str(
            Path(self.index_root)
            / Path(self.experiment_name)
            / "indexes"
            / index_name
        )

    def _get_or_load_index_context(self, index_name: str, create_if_not_exists: bool = False, config_overrides: Optional[Dict[str, Any]] = None) -> ColBERTConfig:
        if index_name in self.index_configs:
            return self.index_configs[index_name]

        current_index_path = self._get_index_path(index_name)
        self.index_paths[index_name] = current_index_path

        # Determine the root for this specific index's config
        # This root is where ColBERT will look for subdirectories like `plan/` or `ivf/` for the index
        specific_index_config_root = str(Path(self.index_root) / self.experiment_name / "indexes")

        if os.path.exists(Path(current_index_path) / "metadata.json"):
            # Index exists, load its specific configuration and data
            if self.verbose > 0:
                print(f"Loading existing index context for '{index_name}' from {current_index_path}")

            # Load the ColBERTConfig specific to this index from its metadata.json
            # This config might have different `doc_maxlen`, `nbits`, etc.
            loaded_config = ColBERTConfig.load_from_index(current_index_path)

            # Ensure its root and experiment are set correctly for operations
            loaded_config.root = specific_index_config_root
            loaded_config.experiment = self.experiment_name
            loaded_config.index_name = index_name # Ensure index_name is part of its config

            self.index_configs[index_name] = loaded_config

            self.model_indices[index_name] = ModelIndexFactory.load_from_file(
                current_index_path, index_name, loaded_config, verbose=self.verbose > 0
            )
            self._get_collection_files_from_disk(index_name, current_index_path)
            return loaded_config
        elif create_if_not_exists:
            if self.verbose > 0:
                print(f"Creating new index context for '{index_name}' at {current_index_path}")
            # Index does not exist, create new context (typically during .index() call)
            # Start with a copy of the base model config, then apply overrides
            new_config = ColBERTConfig.from_existing(self.base_model_config, ColBERTConfig(**(config_overrides or {})))
            new_config.root = specific_index_config_root
            new_config.experiment = self.experiment_name
            new_config.index_name = index_name # Important for operations

            self.index_configs[index_name] = new_config
            self.collections[index_name] = []
            self.pid_docid_maps[index_name] = {}
            self.docid_pid_maps[index_name] = defaultdict(list)
            self.docid_metadata_maps[index_name] = None
            self.model_indices[index_name] = None # Will be created by .index()
            return new_config
        else:
            raise FileNotFoundError(f"Index '{index_name}' not found at {current_index_path} and create_if_not_exists is False.")

    def _invert_pid_docid_map(self, index_name: str) -> Dict[str, List[int]]:
        d = defaultdict(list)
        if index_name not in self.pid_docid_maps:
             if self.verbose > 0:
                print(f"Warning: pid_docid_map not found for index '{index_name}' during inversion.")
             return d
        for k, v in self.pid_docid_maps[index_name].items():
            d[v].append(k)
        return d

    def _get_collection_files_from_disk(self, index_name: str, index_path_str: str):
        index_path = Path(index_path_str)
        self.collections[index_name] = srsly.read_json(index_path / "collection.json")

        if os.path.exists(index_path / "docid_metadata_map.json"):
            self.docid_metadata_maps[index_name] = srsly.read_json(
                index_path / "docid_metadata_map.json"
            )
        else:
            self.docid_metadata_maps[index_name] = None

        try:
            pid_docid_map_data = srsly.read_json(index_path / "pid_docid_map.json")
        except FileNotFoundError as err:
            raise FileNotFoundError(
                f"ERROR: Could not load pid_docid_map.json for index '{index_name}' from {index_path}!",
                "This is likely because you are loading an older, incompatible index or the index is corrupted.",
            ) from err

        self.pid_docid_maps[index_name] = {
            int(key): value for key, value in pid_docid_map_data.items()
        }
        self.docid_pid_maps[index_name] = self._invert_pid_docid_map(index_name)

    def _write_collection_files_to_disk(self, index_name: str):
        if index_name not in self.index_paths:
            raise ValueError(f"Index path for '{index_name}' not found. Ensure index is initialized.")
        current_index_path = self.index_paths[index_name]

        os.makedirs(current_index_path, exist_ok=True)

        srsly.write_json(Path(current_index_path) / "collection.json", self.collections.get(index_name, []))
        srsly.write_json(Path(current_index_path) / "pid_docid_map.json", self.pid_docid_maps.get(index_name, {}))
        if self.docid_metadata_maps.get(index_name) is not None:
            srsly.write_json(
                Path(current_index_path) / "docid_metadata_map.json", self.docid_metadata_maps[index_name]
            )
        self.docid_pid_maps[index_name] = self._invert_pid_docid_map(index_name)


    def _save_index_metadata(self, index_name: str):
        config = self._get_or_load_index_context(index_name)
        model_index = self.model_indices.get(index_name)
        if model_index is None:
            # This can happen if _save_index_metadata is called before ModelIndex is fully constructed (e.g. during .index())
            # In such cases, the ModelIndex construction itself will save its metadata.
            # However, if we are updating metadata for an *existing* model_index, it must be present.
            # Let's check if the index path exists, implying a ModelIndex should have been loaded or is being built.
            if not os.path.exists(self.index_paths[index_name]):
                 raise ValueError(f"ModelIndex for '{index_name}' not found. Cannot save metadata.")
            if self.verbose > 0:
                print(f"ModelIndex for '{index_name}' not yet available for metadata export, assuming it will be handled by ModelIndex.construct.")
            # We still need to save RAGatouille specific files if they exist
            self._write_collection_files_to_disk(index_name)
            return


        current_index_path = self.index_paths[index_name]
        os.makedirs(current_index_path, exist_ok=True) # Ensure directory exists

        metadata_file_path = Path(current_index_path) / "metadata.json"

        model_metadata = {}
        if metadata_file_path.exists():
            try:
                model_metadata = srsly.read_json(str(metadata_file_path))
            except ValueError: # Handles empty or malformed JSON
                if self.verbose > 0:
                    print(f"Warning: metadata.json for index '{index_name}' was malformed or empty. Creating a new one.")
                model_metadata = {}


        index_specific_config = model_index.export_metadata()
        index_specific_config["index_name"] = index_name # Ensure index_name is part of the stored metadata

        # Merge ColBERTConfig parameters into the RAGatouille specific metadata for completeness
        # These are parameters like nbits, doc_maxlen etc. specific to this index

        # Explicitly list relevant, serializable ColBERTConfig fields to store.
        # This avoids trying to serialize complex objects or functions that might be part of ColBERTConfig's internal state.
        relevant_colbert_config_fields = [
            "nbits",
            "doc_maxlen",
            "query_maxlen",
            "index_bsize",
            "kmeans_niters",
            "gpus", # Number of GPUs used for this index creation/config
            "bsize", # General batch size, distinct from index_bsize
            "similarity", # e.g., 'cosine' or 'l2'
            # Add other simple, serializable config fields as deemed necessary
            # Be cautious not to include fields that are functions, complex objects,
            # or already managed (like root, experiment, index_name, checkpoint).
        ]
        config_to_store = {}
        for field_name in relevant_colbert_config_fields:
            if hasattr(config, field_name):
                config_to_store[field_name] = getattr(config, field_name)

        ragatouille_metadata = {
            "index_config": index_specific_config, # From ModelIndex.export_metadata()
            "colbert_config_params": config_to_store,
        }

        model_metadata["RAGatouille"] = ragatouille_metadata
        srsly.write_json(str(metadata_file_path), model_metadata)
        self._write_collection_files_to_disk(index_name)


    def index(
        self,
        index_name: str,
        collection: List[str],
        pid_docid_map: Dict[int, str],
        docid_metadata_map: Optional[dict] = None,
        max_document_length: int = 256,
        overwrite: Union[bool, str] = "reuse",
        bsize: int = 32,
        use_faiss: bool = False,
        **kwargs, # For additional ColBERTConfig settings for this specific index
    ):
        config_overrides = kwargs
        config_overrides['doc_maxlen'] = max_document_length
        config_overrides['index_bsize'] = bsize
        # nbits is determined inside PLAIDModelIndex.build based on collection size

        config = self._get_or_load_index_context(index_name, create_if_not_exists=True, config_overrides=config_overrides)
        # config is now the specific config for index_name

        self.collections[index_name] = collection
        self.pid_docid_maps[index_name] = pid_docid_map
        self.docid_pid_maps[index_name] = self._invert_pid_docid_map(index_name)
        self.docid_metadata_maps[index_name] = docid_metadata_map

        # The ModelIndexFactory will use config.root, config.experiment, and index_name
        # to determine the actual path for storing index files (e.g. IVF lists, centroids)
        # config.root is already Path(self.index_root) / self.experiment_name / "indexes"
        # So the factory will create files under config.root / index_name / ...

        model_index_instance = ModelIndexFactory.construct(
            index_type="PLAID", # Currently PLAID is the main supported type
            config=config, # Pass the specific config for this index
            checkpoint=str(self.base_pretrained_model_name_or_path),
            collection=self.collections[index_name],
            index_name=index_name, # This is crucial
            overwrite=overwrite,
            verbose=self.verbose > 0,
            bsize=bsize,
            use_faiss=use_faiss,
        )
        self.model_indices[index_name] = model_index_instance
        # Update self.index_configs[index_name] with any modifications made by ModelIndexFactory (e.g. nbits)
        self.index_configs[index_name] = model_index_instance.config

        self._save_index_metadata(index_name)

        if self.verbose > 0:
            print(f"Done indexing for index '{index_name}'!")
        return self.index_paths[index_name]


    def add_to_index(
        self,
        index_name: str,
        new_documents: List[str], # These are chunks/passages
        new_pid_docid_map_for_new_docs: Dict[int, str], # PIDs here are 0-indexed for new_documents
        new_docid_metadata_map: Optional[Dict[str, Any]] = None, # Metadata for document_ids in new_documents
        bsize: int = 32,
        use_faiss: bool = False, # Relevant if index needs to be rebuilt
    ):
        config = self._get_or_load_index_context(index_name, create_if_not_exists=False)
        # config.root is Path(self.index_root) / self.experiment_name / "indexes"
        # The model_index will operate within its specific directory: config.root / index_name

        if self.verbose > 0:
            print(
            f"WARNING: add_to_index support for index '{index_name}' is currently experimental!",
            "add_to_index support will be more thorough in future versions",
            )

        current_collection = self.collections.get(index_name, [])
        current_pid_docid_map = self.pid_docid_maps.get(index_name, {})
        # Ensure current_docid_metadata_map is a dict, not None, before .update()
        current_docid_metadata_map_val = self.docid_metadata_maps.get(index_name)
        if current_docid_metadata_map_val is None:
            current_docid_metadata_map = {}
        else:
            current_docid_metadata_map = current_docid_metadata_map_val


        # Filter out documents whose document_ids are already present
        # This assumes new_pid_docid_map_for_new_docs gives the final intended document_id for each new passage
        existing_doc_ids_in_index = set(current_pid_docid_map.values())

        truly_new_passages_with_original_pid = [] # Stores (original_pid_in_new_docs, passage_content)
        for original_pid, passage_content in enumerate(new_documents):
            doc_id_for_this_passage = new_pid_docid_map_for_new_docs.get(original_pid)
            if doc_id_for_this_passage is None:
                if self.verbose > 0:
                    print(f"Warning: Original PID {original_pid} in new_documents not found in new_pid_docid_map_for_new_docs. Skipping.")
                continue
            if doc_id_for_this_passage not in existing_doc_ids_in_index:
                truly_new_passages_with_original_pid.append((original_pid, passage_content, doc_id_for_this_passage))

        if not truly_new_passages_with_original_pid:
            if self.verbose > 0:
                print(f"No new unique documents to add to index '{index_name}'. All provided document IDs already exist or no valid new documents provided.")
            return

        new_collection_for_indexer = [item[1] for item in truly_new_passages_with_original_pid]

        # Update persistent maps
        max_existing_pid = max(current_pid_docid_map.keys(), default=-1)
        for idx, (original_pid, passage_content, doc_id_for_this_passage) in enumerate(truly_new_passages_with_original_pid):
            new_global_pid = max_existing_pid + 1 + idx
            current_pid_docid_map[new_global_pid] = doc_id_for_this_passage

        current_collection.extend(new_collection_for_indexer)

        if new_docid_metadata_map:
            # current_docid_metadata_map is guaranteed to be a dict here
            current_docid_metadata_map.update(new_docid_metadata_map)
        
        self.collections[index_name] = current_collection
        self.pid_docid_maps[index_name] = current_pid_docid_map
        self.docid_metadata_maps[index_name] = current_docid_metadata_map
        self.docid_pid_maps[index_name] = self._invert_pid_docid_map(index_name)

        model_index = self.model_indices.get(index_name)
        if model_index is None:
            raise ValueError(f"ModelIndex for '{index_name}' not found. Cannot add to it. Has it been indexed first?")

        model_index.add(
            config=config, # Pass the specific config
            checkpoint=str(self.base_pretrained_model_name_or_path),
            collection=self.collections[index_name], # Pass the full current collection
            index_root=config.root, # This is like Path(self.index_root) / self.experiment_name / "indexes"
            index_name=index_name, # Crucial
            new_collection=new_collection_for_indexer, # Only the truly new passages
            verbose=self.verbose > 0,
            bsize=bsize,
            use_faiss=use_faiss, # For potential rebuild
        )
        self.index_configs[index_name] = model_index.config # Update with potentially modified config from .add()

        self._save_index_metadata(index_name)

        if self.verbose > 0:
            print(
                f"Successfully attempted to update index '{index_name}' with {len(new_collection_for_indexer)} new passages!\n",
                f"New index size for '{index_name}': {len(self.collections[index_name])} passages.",
            )


    def delete_from_index(
        self,
        index_name: str,
        document_ids: Union[TypeVar("T"), List[TypeVar("T")]], # Document IDs to remove
    ):
        config = self._get_or_load_index_context(index_name, create_if_not_exists=False)

        if self.verbose > 0:
            print(
                f"WARNING: delete_from_index support for index '{index_name}' is currently experimental!",
                "delete_from_index support will be more thorough in future versions",
            )

        current_collection = self.collections.get(index_name)
        current_pid_docid_map = self.pid_docid_maps.get(index_name)
        current_docid_metadata_map = self.docid_metadata_maps.get(index_name)

        if not current_collection or not current_pid_docid_map:
            if self.verbose > 0:
                print(f"Index '{index_name}' appears to be empty or not loaded. Cannot delete.")
            return

        pids_to_remove_from_indexer = []
        doc_ids_to_remove_set = set(document_ids if isinstance(document_ids, list) else [document_ids])

        new_pid_docid_map = {}
        kept_passage_indices = [] # 0-indexed relative to current_collection

        # Iterate through current PIDs to find which ones to remove
        # And construct the list of PIDs for the Indexer.remove() method
        for pid_key, docid_val in current_pid_docid_map.items():
            if docid_val in doc_ids_to_remove_set:
                pids_to_remove_from_indexer.append(pid_key)
            else:
                # This passage is kept. We need its content for the new collection.
                # The pid_key is its current global PID, which is also its index in current_collection.
                # We need to map this old pid_key to a new pid_key if PIDs are compacted later.
                # For now, just track which passages are kept.
                # The indexer.remove() takes global PIDs.
                pass # Handled by the model_index.delete itself based on pids_to_remove_from_indexer

        if not pids_to_remove_from_indexer:
            if self.verbose > 0:
                print(f"No documents corresponding to the given IDs found in index '{index_name}'. Nothing to delete.")
            return

        model_index = self.model_indices.get(index_name)
        if model_index is None:
            raise ValueError(f"ModelIndex for '{index_name}' not found. Cannot delete. Has it been indexed first?")

        model_index.delete(
            config=config,
            checkpoint=str(self.base_pretrained_model_name_or_path),
            collection=current_collection, # Pass the full current collection
            index_name=index_name,
            pids_to_remove=pids_to_remove_from_indexer,
            verbose=self.verbose > 0,
        )
        self.index_configs[index_name] = model_index.config


        # After indexer has processed deletions, update our local high-level trackings
        # Rebuild collections, pid_docid_maps, etc. based on what remains
        # This is simpler than trying to surgically remove and re-index PIDs.
        # The indexer internally handles tombstones or compaction.
        # For our metadata, we effectively "reload" it reflecting the deletions.

        # A bit inefficient, but safest: re-filter based on the pids *not* removed
        new_collection_content = []
        new_pid_to_docid = {}
        next_new_pid = 0
        for old_pid, passage_content in enumerate(current_collection):
            if old_pid not in pids_to_remove_from_indexer: # If this passage was NOT deleted by indexer
                new_collection_content.append(passage_content)
                # Map its original document_id to the new PID
                original_doc_id = current_pid_docid_map.get(old_pid)
                if original_doc_id: # Should always exist
                    new_pid_to_docid[next_new_pid] = original_doc_id
                next_new_pid += 1

        self.collections[index_name] = new_collection_content
        self.pid_docid_maps[index_name] = new_pid_to_docid

        if current_docid_metadata_map is not None:
            self.docid_metadata_maps[index_name] = {
                docid: metadata
                for docid, metadata in current_docid_metadata_map.items()
                if docid not in doc_ids_to_remove_set
            }

        self.docid_pid_maps[index_name] = self._invert_pid_docid_map(index_name)
        self._save_index_metadata(index_name)

        if self.verbose > 0:
            print(f"Successfully deleted documents with IDs {doc_ids_to_remove_set} from index '{index_name}'.")
            print(f"New index size for '{index_name}': {len(self.collections[index_name])} passages.")


    def search(
        self,
        index_name: str,
        query: Union[str, list[str]],
        k: int = 10,
        force_fast: bool = False,
        zero_index_ranks: bool = False,
        doc_ids: Optional[List[str]] = None, # Filter search to these document_ids
    ):
        print("1")
        config = self._get_or_load_index_context(index_name, create_if_not_exists=False)
        model_index = self.model_indices.get(index_name)
        current_collection = self.collections.get(index_name)
        current_pid_docid_map = self.pid_docid_maps.get(index_name)
        current_docid_pid_map = self.docid_pid_maps.get(index_name)
        current_docid_metadata_map = self.docid_metadata_maps.get(index_name)
        print("2", current_pid_docid_map)

        if model_index is None or not current_collection or not current_pid_docid_map or not current_docid_pid_map:
            raise ValueError(f"Index '{index_name}' is not properly loaded or is empty. Cannot search.")
        print("3")
        pids_to_search_within = None
        if doc_ids is not None:
            pids_to_search_within = []
            for doc_id_filter in doc_ids:
                pids_to_search_within.extend(current_docid_pid_map.get(doc_id_filter, []))
            if not pids_to_search_within:
                if self.verbose > 0:
                    print(f"Warning: doc_ids filter provided for index '{index_name}', but no passages found for these doc_ids. Returning empty results.")
                return [] if isinstance(query, str) else [[] for _ in query]
        print("4")
        # The model_index search needs the full collection context if it's not already loaded by its Searcher
        # However, PLAIDModelIndex typically loads its own collection view.
        results_from_model_index = model_index.search(
            config=config,
            checkpoint=str(self.base_pretrained_model_name_or_path),
            collection=current_collection, # Provided for context, Searcher might use its own view
            index_name=index_name,
            base_model_max_tokens=self.base_model_max_tokens,
            query=query,
            k=k,
            pids=pids_to_search_within, # Pass the filtered PIDs to the searcher
            force_reload=False, # We handle context loading; force_reload here is for searcher specific state.
            force_fast=force_fast,
        )
        print("5")
        to_return = []
        for result_group in results_from_model_index: # result_group is for one query
            result_for_query = []
            # result_group is typically (passage_ids, ranks, scores)
            passage_pids, ranks, scores = result_group[0], result_group[1], result_group[2]

            for passage_pid, rank_val, score_val in zip(passage_pids, ranks, scores):
                document_id_for_passage = current_pid_docid_map.get(int(passage_pid))
                if document_id_for_passage is None:
                    if self.verbose > 0:
                        print(f"Warning: Passage PID {passage_pid} from search results not found in pid_docid_map for index '{index_name}'. Skipping.")
                    continue

                # Ensure passage_pid is valid index for current_collection
                if not (0 <= int(passage_pid) < len(current_collection)):
                    if self.verbose > 0:
                        print(f"Warning: Passage PID {passage_pid} from search results is out of bounds for current_collection of index '{index_name}'. Skipping.")
                    continue


                result_dict = {
                    "content": current_collection[int(passage_pid)],
                    "score": float(score_val),
                    "rank": int(rank_val) - 1 if zero_index_ranks else int(rank_val),
                    "document_id": document_id_for_passage,
                    "passage_id": int(passage_pid), # This is the internal PID used by ColBERT
                }

                if current_docid_metadata_map is not None:
                    if document_id_for_passage in current_docid_metadata_map:
                        doc_metadata = current_docid_metadata_map[document_id_for_passage]
                        result_dict["document_metadata"] = doc_metadata

                result_for_query.append(result_dict)
            to_return.append(result_for_query)

        return to_return[0] if isinstance(query, str) and len(to_return) == 1 else to_return


    # Methods for index-free operations (rank, encode, search_encoded_docs)
    # These operate on the base model and don't interact with specific disk-based indices.
    # They use self.inference_ckpt directly.

    def _set_inference_max_tokens(
        self, documents: list[str], max_tokens: Union[Literal["auto"], int] = "auto"
    ):
        # This state should be temporary per call for index-free, or managed if we cache encodings
        if not hasattr(self, "_temp_inference_ckpt_len_set") or self._temp_inference_ckpt_len_set is False:
            if max_tokens == "auto":
                # Calculate based on documents, ensuring it doesn't exceed base_model_max_tokens
                percentile_90 = np.percentile([len(x.split(" ")) for x in documents], 90)
                effective_max_tokens = min(
                    math.floor((math.ceil((percentile_90 * 1.35) / 32) * 32) * 1.1),
                    self.base_model_max_tokens,
                )
                effective_max_tokens = max(256, int(effective_max_tokens)) # Ensure a minimum reasonable length
            else:
                effective_max_tokens = min(int(max_tokens), self.base_model_max_tokens)

            if effective_max_tokens > 300 and self.verbose > 0 and max_tokens == "auto":
                 print(
                    f"Your documents are roughly {percentile_90} tokens long at the 90th percentile for this batch!",
                    "This is quite long and might slow down reranking!\n",
                    "Provide fewer documents, build smaller chunks or run on GPU if it takes too long for your needs!",
                )

            # Temporarily adjust the main inference_ckpt for this operation
            # A more robust solution might involve creating a temporary tokenizer instance
            self.inference_ckpt.colbert_config.doc_maxlen = effective_max_tokens
            self.inference_ckpt.doc_tokenizer.doc_maxlen = effective_max_tokens
            self._temp_inference_ckpt_len_set = True # Mark that we've set it for this call scope


    def _reset_inference_max_tokens(self):
        # Reset to base model's default after the operation
        if hasattr(self, "_temp_inference_ckpt_len_set") and self._temp_inference_ckpt_len_set:
            base_doc_maxlen = self.inference_ckpt.bert.config.max_position_embeddings - 4 # Re-calculate default
            self.inference_ckpt.colbert_config.doc_maxlen = base_doc_maxlen
            self.inference_ckpt.doc_tokenizer.doc_maxlen = base_doc_maxlen
            del self._temp_inference_ckpt_len_set


    def _colbert_score(self, Q, D_padded, D_mask):
        # Standard ColBERT scoring logic
        if ColBERTConfig().total_visible_gpus > 0: # Check global config for GPU availability
            Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

        assert Q.dim() == 3, Q.size()
        assert D_padded.dim() == 3, D_padded.size()
        assert Q.size(0) in [1, D_padded.size(0)]

        scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)
        scores = scores.max(1).values
        return scores.sum(-1)

    def _encode_index_free_queries(
        self,
        queries: Union[str, list[str]],
        bsize: Union[Literal["auto"], int] = "auto",
    ):
        if bsize == "auto":
            bsize = 32
        if isinstance(queries, str):
            queries = [queries]

        # Temporarily adjust query_maxlen for this batch of queries
        original_query_maxlen = self.inference_ckpt.query_tokenizer.query_maxlen
        maxlen = max([int(len(x.split(" ")) * 1.35) for x in queries])
        self.inference_ckpt.query_tokenizer.query_maxlen = max(
            min(maxlen, self.base_model_max_tokens), 32 # Maximize length but not over model capacity
        )

        embedded_queries = [
            x.unsqueeze(0)
            for x in self.inference_ckpt.queryFromText(queries, bsize=bsize, to_cpu=True) # Ensure CPU for general use
        ]

        self.inference_ckpt.query_tokenizer.query_maxlen = original_query_maxlen # Restore
        return embedded_queries

    def _encode_index_free_documents(
        self,
        documents: list[str],
        bsize: Union[Literal["auto"], int] = "auto",
        verbose_override: Optional[bool] = None,
    ):
        effective_verbose = self.verbose > 0 if verbose_override is None else verbose_override

        # bsize calculation needs to consider the currently set doc_maxlen on inference_ckpt
        current_doc_maxlen_for_bsize_calc = self.inference_ckpt.doc_tokenizer.doc_maxlen
        if bsize == "auto":
            bsize = 32
            if current_doc_maxlen_for_bsize_calc > 512: # Example threshold
                # Reduce bsize proportionally if doc_maxlen is very large to avoid OOM
                factor = 2 ** round(math.log(current_doc_maxlen_for_bsize_calc / 512, 2))
                bsize = max(1, int(32 / factor))
                if effective_verbose:
                    print(f"Auto bsize adjusted to {bsize} due to large doc_maxlen ({current_doc_maxlen_for_bsize_calc})")

        # D_encoder returns (tensor, count). We only need tensor.
        # to_cpu=True to ensure results are on CPU if not using GPU for ColBERT overall
        embedded_docs_tensor = self.inference_ckpt.docFromText(
            documents, bsize=bsize, showprogress=effective_verbose, to_cpu=True
        )[0]

        # Mask is usually for padding in ColBERT, here we create a full mask assuming all tokens are valid for scoring.
        # This might need adjustment if specific masking strategies are required for index-free.
        # For simple maxsim, the mask might not be strictly necessary if D_padded already handles lengths.
        # However, _colbert_score expects D_mask.
        doc_mask = torch.zeros(embedded_docs_tensor.shape[:2], device=embedded_docs_tensor.device) # All valid

        return embedded_docs_tensor, doc_mask


    def _index_free_search(
        self,
        embedded_queries, # List of query tensors
        documents: list[str], # Original documents for retrieval
        embedded_docs, # Tensor of document embeddings
        doc_mask, # Document masks
        k: int = 10,
        zero_index_ranks: bool = False,
    ):
        results_all_queries = []

        for query_tensor in embedded_queries: # query_tensor is [1, num_tokens, dim]
            scores = self._colbert_score(query_tensor, embedded_docs, doc_mask) # scores will be [num_docs]

            # Ensure k is not larger than number of documents
            effective_k = min(k, embedded_docs.size(0))

            top_k_scores, top_k_indices = torch.topk(scores, effective_k)

            results_for_single_query = []
            for rank, doc_idx_in_batch in enumerate(top_k_indices.tolist()):
                result = {
                    "content": documents[doc_idx_in_batch],
                    "score": float(scores[doc_idx_in_batch]), # Get original score before topk for consistency
                    "rank": rank if zero_index_ranks else rank + 1,
                    "result_index": doc_idx_in_batch, # Index within the provided documents list
                }
                results_for_single_query.append(result)
            results_all_queries.append(results_for_single_query)

        return results_all_queries[0] if len(results_all_queries) == 1 else results_all_queries


    def rank(
        self,
        query: str, # Single query string for ranking
        documents: list[str],
        k: int = 10,
        zero_index_ranks: bool = False,
        bsize: Union[Literal["auto"], int] = "auto",
        max_tokens: Union[Literal["auto"], int] = "auto",
    ):
        if not documents:
            return []
        if k > len(documents):
            if self.verbose > 0:
                print(f"Warning: k value ({k}) for rank is larger than the number of documents ({len(documents)}). Adjusting k.")
            k = len(documents)

        try:
            self._set_inference_max_tokens(documents=documents, max_tokens=max_tokens)

            # Encode query
            embedded_queries = self._encode_index_free_queries([query], bsize=bsize) # List with one query tensor

            # Encode documents
            # verbose_override=False for document encoding within rank unless self.verbose is high
            embedded_docs, doc_mask = self._encode_index_free_documents(documents, bsize=bsize, verbose_override=(self.verbose > 1))

            # Perform search (ranking)
            # _index_free_search returns list of lists; since it's one query, take first element
            ranked_results = self._index_free_search(
                embedded_queries, documents, embedded_docs, doc_mask, k, zero_index_ranks
            )
            return ranked_results # This will be the list of dicts for the single query
        finally:
            self._reset_inference_max_tokens()


    # --- Methods for managing cached encoded documents (experimental) ---
    # These would allow encoding a large set once and searching over it multiple times.
    def encode(
        self,
        documents: list[str],
        document_metadatas: Optional[List[Optional[Dict[str, Any]]]] = None, # Optional metadata per document
        bsize: int = 32,
        max_tokens: Union[Literal["auto"], int] = "auto",
        verbose_override: Optional[bool] = None,
    ):
        """Encodes documents and stores them in memory for subsequent `search_encoded_docs` calls."""
        if not hasattr(self, "_cached_encodings"):
            self._cached_encodings = {
                "collection": [],
                "metadatas": [],
                "embeddings": None, # Will be a tensor
                "masks": None, # Will be a tensor
                "doc_maxlen_at_encoding": 0,
            }

        effective_verbose = self.verbose > 0 if verbose_override is None else verbose_override
        try:
            self._set_inference_max_tokens(documents=documents, max_tokens=max_tokens)

            # Store the doc_maxlen used for this batch, important if concatenating
            current_encoding_doc_maxlen = self.inference_ckpt.doc_tokenizer.doc_maxlen

            new_embeddings, new_masks = self._encode_index_free_documents(
                documents, bsize=bsize, verbose_override=effective_verbose
            )

            # Normalize embedding shapes for concatenation if doc_maxlen changed
            if self._cached_encodings["embeddings"] is not None:
                # If cached encodings exist, all new encodings must match its doc_maxlen
                # Or, more robustly, pad all to the largest encountered doc_maxlen
                # For now, let's assume subsequent encodes try to match or clear first
                if current_encoding_doc_maxlen != self._cached_encodings["doc_maxlen_at_encoding"]:
                    # This logic needs to be robust: pad existing or new to match a common dimension.
                    # Simplest for now: raise error or warn, require consistent max_tokens or clear cache.
                    print(f"Warning: max_tokens for this encode call ({current_encoding_doc_maxlen}) differs from cached ({self._cached_encodings['doc_maxlen_at_encoding']}). Results may be inconsistent if concatenating. Consider clearing cache or using consistent max_tokens.")
                    # Fallback: Pad new_embeddings to match cached dimension if smaller, or re-encode all if larger (costly)
                    # This padding ensures self._colbert_score doesn't fail on shape mismatch.
                    if new_embeddings.shape[1] < self._cached_encodings["doc_maxlen_at_encoding"]:
                        padding_size = self._cached_encodings["doc_maxlen_at_encoding"] - new_embeddings.shape[1]
                        pad_tensor = torch.zeros(new_embeddings.shape[0], padding_size, new_embeddings.shape[2], device=new_embeddings.device)
                        new_embeddings = torch.cat([new_embeddings, pad_tensor], dim=1)
                        # Also pad masks if they are length-dependent (current D_mask from _encode_index_free is not, but good practice)

            self._cached_encodings["collection"].extend(documents)
            if document_metadatas:
                self._cached_encodings["metadatas"].extend(document_metadatas)
            else:
                self._cached_encodings["metadatas"].extend([None] * len(documents))

            if self._cached_encodings["embeddings"] is None:
                self._cached_encodings["embeddings"] = new_embeddings
                self._cached_encodings["masks"] = new_masks
                self._cached_encodings["doc_maxlen_at_encoding"] = current_encoding_doc_maxlen
            else:
                # Ensure device consistency before cat
                device = self._cached_encodings["embeddings"].device
                self._cached_encodings["embeddings"] = torch.cat(
                    [self._cached_encodings["embeddings"], new_embeddings.to(device)], dim=0
                )
                self._cached_encodings["masks"] = torch.cat(
                    [self._cached_encodings["masks"], new_masks.to(device)], dim=0
                )
            if effective_verbose:
                print(f"Encoded {len(documents)} documents. Total cached: {len(self._cached_encodings['collection'])}.")

        finally:
            self._reset_inference_max_tokens()


    def search_encoded_docs(
        self,
        queries: Union[str, list[str]],
        k: int = 10,
        bsize: int = 32, # For query encoding
        zero_index_ranks: bool = False,
    ):
        if not hasattr(self, "_cached_encodings") or self._cached_encodings["embeddings"] is None:
            print("No documents have been encoded and cached. Call .encode() first.")
            return [] if isinstance(queries, str) else [[] for _ in queries]

        # Encode queries (max_tokens for queries is handled by _encode_index_free_queries)
        embedded_queries = self._encode_index_free_queries(queries, bsize=bsize)

        results = self._index_free_search(
            embedded_queries,
            self._cached_encodings["collection"],
            self._cached_encodings["embeddings"],
            self._cached_encodings["masks"],
            k,
            zero_index_ranks,
        )

        # Add metadata if available
        # results is either a list of dicts (single query) or list of lists of dicts (multi-query)
        def add_meta(single_query_results_list):
            for r_dict in single_query_results_list:
                doc_meta = self._cached_encodings["metadatas"][r_dict["result_index"]]
                if doc_meta:
                    r_dict["document_metadata"] = doc_meta
            return single_query_results_list

        if isinstance(queries, str): # Single query, results is a list of dicts
            return add_meta(results)
        else: # Multiple queries, results is a list of lists of dicts
            return [add_meta(res_list) for res_list in results]


    def clear_encoded_docs(self, force: bool = False):
        if not hasattr(self, "_cached_encodings") or self._cached_encodings["embeddings"] is None:
            print("No cached encodings to clear.")
            return

        if not force:
            if self.verbose > 0:
                print("All in-memory encodings will be deleted in 5 seconds. Interrupt to cancel.")
            time.sleep(5)

        self._cached_encodings = {
            "collection": [], "metadatas": [], "embeddings": None, "masks": None, "doc_maxlen_at_encoding":0
        }
        if torch.cuda.is_available(): # Try to free GPU memory if used
            torch.cuda.empty_cache()
        if self.verbose > 0:
            print("Cached encoded documents cleared.")


    # --- Training method (largely unchanged, uses global RunContext) ---
    def train(self, index_name_for_config: Optional[str], data_dir: Union[str, Path], training_config_overrides: Dict[str, Any]) -> str:
        # Training often creates a new model/checkpoint.
        # It needs a ColBERTConfig. If an index_name is provided, use its config as a base.
        # Otherwise, use the base_model_config.
        if index_name_for_config:
            base_train_config = self._get_or_load_index_context(index_name_for_config, create_if_not_exists=False) # Should exist
        else:
            base_train_config = self.base_model_config # Default if no specific index config

        # Construct ColBERTConfig from the dictionary of overrides
        override_config_obj = ColBERTConfig(**training_config_overrides)
        final_training_config = ColBERTConfig.from_existing(base_train_config, override_config_obj)
        final_training_config.nway = 2 # Common setting for ColBERT training

        data_path = Path(data_dir)

        # Run().context(self.run_config) is already active globally for the class instance
        trainer = Trainer(
            triples=str(data_path / "triples.train.colbert.jsonl"),
            queries=str(data_path / "queries.train.colbert.tsv"),
            collection=str(data_path / "corpus.train.colbert.tsv"),
            config=final_training_config, # Pass the resolved training config
        )
        # Trainer uses the checkpoint from the ColBERTConfig if not specified otherwise
        trainer.train(checkpoint=str(self.base_pretrained_model_name_or_path))

        if self.verbose > 0:
            print(f"Training complete. Best checkpoint saved to: {trainer.path_}") # trainer.path_ is new standard
        return trainer.path_


    def __del__(self):
        # Clean up global ColBERT context
        try:
            if hasattr(self, 'run_context') and self.run_context:
                self.run_context.__exit__(None, None, None)
        except Exception:
            if self.verbose > 0: # Only print if verbose
                print("INFO: Tried to clean up ColBERT RunContext but failed. This is usually not critical.")

        # Clear cached encodings if any, to free memory
        if hasattr(self, "_cached_encodings"):
            del self._cached_encodings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
