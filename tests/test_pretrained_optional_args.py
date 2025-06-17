import os
import shutil
from pathlib import Path

import pytest
import srsly

from ragatouille import RAGPretrainedModel

# Sample data for tests
COLLECTION_SAMPLES = [
    "Hayao Miyazaki is a Japanese animator, filmmaker, and manga artist. He co-founded Studio Ghibli.",
    "Studio Ghibli, Inc. is a Japanese animation studio based in Koganei, Tokyo. Its mascot is Totoro.",
    "Princess Mononoke is a 1997 Japanese animated epic historical fantasy film by Studio Ghibli.",
]

DOCUMENT_IDS_SAMPLES = ["miyazaki_bio", "ghibli_info", "mononoke_film"]

DOCUMENT_METADATAS_SAMPLES = [
    {"entity": "person", "source": "wikipedia", "year": 1941},
    {"entity": "organisation", "source": "wikipedia", "founded": 1985},
    {"entity": "film", "source": "wikipedia", "release_year": 1997},
]

# Use a fixed model name for tests for consistency
PRETRAINED_MODEL_NAME = "colbert-ir/colbertv2.0"
TEST_EXPERIMENT_NAME = "ragatouille_test_suite_optional_args"


@pytest.fixture(scope="session")
def persistent_test_index_root(tmp_path_factory):
    # Creates a temporary root directory for all indices for the test session
    path = tmp_path_factory.mktemp("ragatouille_test_indices_root_optional_args")
    yield str(path)
    # Cleanup after session
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module")  # One RAG model instance per test module (this file)
def rag_model_instance(persistent_test_index_root):
    return RAGPretrainedModel.from_pretrained(
        PRETRAINED_MODEL_NAME,
        index_root=persistent_test_index_root,
        experiment_name=TEST_EXPERIMENT_NAME,
        verbose=0  # Keep test output clean
    )


# Parameterize test cases for different combinations of inputs to RAG.index()
INDEX_CREATION_PARAMS = [
    pytest.param(
        {
            "documents": COLLECTION_SAMPLES[:1],
            "document_ids": None, # Test auto-generation
            "document_metadatas": None,
            "index_name_suffix": "no_opts_auto_ids",
            "max_document_length": 128,
        },
        id="no_optional_args_auto_ids",
    ),
    pytest.param(
        {
            "documents": COLLECTION_SAMPLES[:1],
            "document_ids": DOCUMENT_IDS_SAMPLES[:1],
            "document_metadatas": None,
            "index_name_suffix": "with_docid",
            "max_document_length": 512, # Simulate no splitting
        },
        id="with_doc_ids_no_split",
    ),
    pytest.param(
        {
            "documents": COLLECTION_SAMPLES[:2],
            "document_ids": DOCUMENT_IDS_SAMPLES[:2],
            "document_metadatas": DOCUMENT_METADATAS_SAMPLES[:2],
            "index_name_suffix": "with_docid_meta",
            "max_document_length": 64, # Simulate splitting
        },
        id="with_doc_ids_metadata_split",
    ),
    pytest.param(
        {
            "documents": COLLECTION_SAMPLES,
            "document_ids": DOCUMENT_IDS_SAMPLES,
            "document_metadatas": DOCUMENT_METADATAS_SAMPLES,
            "index_name_suffix": "all_docs_all_details",
            "max_document_length": 128,
        },
        id="all_docs_all_details_split",
    ),
]


@pytest.fixture(scope="function", params=INDEX_CREATION_PARAMS)
def index_params_for_function(request, rag_model_instance):
    # Generates a unique index name for each parameterized test function run
    base_params = request.param.copy()
    # Create a unique index name based on test name and params
    # to ensure isolation even if tests run in parallel within the module (though pytest usually serializes by default)
    unique_suffix = request.node.name.replace("test_", "").replace("[", "_").replace("]", "")
    base_params["index_name"] = f"test_idx_{base_params['index_name_suffix']}_{unique_suffix}"

    # Ensure document_ids are correctly handled for RAG.index()
    # If document_ids is None in params, RAGPretrainedModel will auto-generate them.
    # Store the effective document_ids used for assertion later.
    if base_params["document_ids"] is None:
        # RAGPretrainedModel generates UUIDs if None is passed.
        # We can't know them in advance for assertion against pid_docid_map values exactly,
        # but we can check counts and that metadata aligns if generated.
        # For simplicity in this test, we will just ensure metadata isn't expected if IDs were None.
        base_params["effective_document_ids_for_assertion"] = None
    else:
        base_params["effective_document_ids_for_assertion"] = base_params["document_ids"]

    return base_params


@pytest.fixture(scope="function")
def created_index_data(rag_model_instance, index_params_for_function):
    RAG = rag_model_instance
    params = index_params_for_function # Renamed for clarity within this fixture
    index_name = params["index_name"]

    actual_index_path_str = RAG.index(
        index_name=index_name,
        documents=params["documents"],
        document_ids=params["document_ids"], # Pass None if specified in params for auto-generation
        document_metadatas=params["document_metadatas"],
        max_document_length=params["max_document_length"],
        overwrite_index=True # Crucial for test isolation
    )
    yield {
        "index_name": index_name,
        "index_path_str": actual_index_path_str,
        "params_used": params # Contains effective_document_ids_for_assertion
    }
    # Optional: RAG.delete_index(index_name) if it exists, or rely on session cleanup of root.


def get_full_index_path_obj(persistent_test_index_root, index_name):
    return Path(persistent_test_index_root) / TEST_EXPERIMENT_NAME / "indexes" / index_name


def test_index_creation_and_structure(created_index_data, persistent_test_index_root):
    index_name = created_index_data["index_name"]
    params_used = created_index_data["params_used"]
    full_path = get_full_index_path_obj(persistent_test_index_root, index_name)

    assert full_path.exists(), f"Index directory {full_path} was not created."
    assert (full_path / "collection.json").exists(), "collection.json missing."
    assert (full_path / "pid_docid_map.json").exists(), "pid_docid_map.json missing."

    if params_used.get("document_metadatas"):
        assert (full_path / "docid_metadata_map.json").exists(), "docid_metadata_map.json missing when metadata provided."
    else:
        assert not (full_path / "docid_metadata_map.json").exists(), "docid_metadata_map.json exists when no metadata provided."

    pid_docid_map_data = srsly.read_json(str(full_path / "pid_docid_map.json"))
    assert isinstance(pid_docid_map_data, dict)
    
    # Check that number of passages (PIDs) is reasonable
    # If max_document_length is small, expect more passages than original documents
    if params_used["max_document_length"] < 200 and len(params_used["documents"]) > 0 : # Rough check for splitting
        assert len(pid_docid_map_data) >= len(params_used["documents"]), "Splitting should result in more or equal passages than documents"
    elif len(params_used["documents"]) > 0:
         assert len(pid_docid_map_data) == len(params_used["documents"]), "No splitting should result in equal passages and documents"


    if params_used.get("effective_document_ids_for_assertion"):
        provided_doc_ids_set = set(params_used["effective_document_ids_for_assertion"])
        mapped_doc_ids_set = set(pid_docid_map_data.values())
        assert provided_doc_ids_set == mapped_doc_ids_set, "Mismatch between provided document_ids and those in pid_docid_map."


def test_search_results_and_metadata(rag_model_instance, created_index_data):
    RAG = rag_model_instance
    index_name = created_index_data["index_name"]
    params_used = created_index_data["params_used"]

    query = "Hayao Miyazaki Ghibli" # A general query likely to hit sample data
    results = RAG.search(index_name=index_name, query=query, k=1)

    assert isinstance(results, list)
    if params_used["documents"]:
        # Results can be empty if k=0 or no match, so check len >= 0
        assert len(results) <= 1, "Search with k=1 should return at most 1 result."
        if results:
            result = results[0]
            assert "content" in result
            assert "score" in result
            assert "rank" in result
            assert "document_id" in result

            if params_used.get("document_metadatas") and params_used.get("effective_document_ids_for_assertion"):
                assert "document_metadata" in result, "document_metadata missing in search result."
                retrieved_doc_id = result["document_id"]
                try:
                    original_idx = params_used["effective_document_ids_for_assertion"].index(retrieved_doc_id)
                    expected_metadata = params_used["document_metadatas"][original_idx]
                    assert result["document_metadata"] == expected_metadata, "Mismatch in returned document metadata."
                except ValueError:
                    # This can happen if the retrieved_doc_id was auto-generated and not in our effective_document_ids_for_assertion
                    # This part of the check is only fully valid if document_ids were explicitly provided.
                    if params_used["document_ids"] is not None: # Only fail if we expected to find it
                        pytest.fail(f"Retrieved document_id {retrieved_doc_id} not in original document_ids for test.")
            else:
                assert "document_metadata" not in result, "document_metadata present when it was not indexed."
    else: # No documents were indexed
        assert len(results) == 0


def test_add_to_existing_index(rag_model_instance, created_index_data, persistent_test_index_root):
    RAG = rag_model_instance
    index_name = created_index_data["index_name"]
    full_index_dir_path = get_full_index_path_obj(persistent_test_index_root, index_name)
    collection_file_path = full_index_dir_path / "collection.json"
    pid_docid_map_file_path = full_index_dir_path / "pid_docid_map.json"
    docid_metadata_map_file_path = full_index_dir_path / "docid_metadata_map.json"

    initial_collection_len = len(srsly.read_json(str(collection_file_path)))

    new_docs_to_add = ["Toei Animation is another famous Japanese animation studio."]
    new_doc_ids_to_add = ["toei_animation_info_add_test"]
    new_doc_metadatas_to_add = [{"entity": "organisation", "source": "test_add_to_index"}]

    RAG.add_to_index(
        index_name=index_name,
        new_documents=new_docs_to_add,
        new_document_ids=new_doc_ids_to_add,
        new_document_metadatas=new_doc_metadatas_to_add,
    )

    updated_collection = srsly.read_json(str(collection_file_path))
    assert len(updated_collection) > initial_collection_len

    updated_pid_docid_map = srsly.read_json(str(pid_docid_map_file_path))
    assert new_doc_ids_to_add[0] in updated_pid_docid_map.values()

    if docid_metadata_map_file_path.exists() or new_doc_metadatas_to_add: # File may be created if it didn't exist but new metadata is added
        # Wait for file system to catch up if needed, though srsly should handle it.
        if not docid_metadata_map_file_path.exists() and new_doc_metadatas_to_add:
            import time; time.sleep(0.1) # Small delay, usually not needed

        if docid_metadata_map_file_path.exists():
            updated_docid_metadata_map = srsly.read_json(str(docid_metadata_map_file_path))
            assert new_doc_ids_to_add[0] in updated_docid_metadata_map
            assert updated_docid_metadata_map[new_doc_ids_to_add[0]] == new_doc_metadatas_to_add[0]
    
    results_for_new = RAG.search(index_name=index_name, query="Toei Animation studio", k=3)
    found_new_doc_in_search = any(res.get("document_id") == new_doc_ids_to_add[0] for res in results_for_new)
    assert found_new_doc_in_search, "Newly added document not found in search results."


def test_delete_from_index(rag_model_instance, created_index_data, persistent_test_index_root):
    RAG = rag_model_instance
    index_name = created_index_data["index_name"]
    params_used = created_index_data["params_used"]
    
    # This test can only meaningfully run if specific document_ids were used for creation
    if not params_used.get("effective_document_ids_for_assertion"):
        pytest.skip("Skipping delete test as no explicit document_ids were used for this index setup.")

    doc_id_to_delete = params_used["effective_document_ids_for_assertion"][0]
    content_of_deleted_doc = params_used["documents"][0] # Get content to search for it later

    full_index_dir_path = get_full_index_path_obj(persistent_test_index_root, index_name)
    pid_docid_map_file_path = full_index_dir_path / "pid_docid_map.json"
    docid_metadata_map_file_path = full_index_dir_path / "docid_metadata_map.json"

    RAG.delete_from_index(index_name=index_name, document_ids=[doc_id_to_delete])

    updated_pid_docid_map = srsly.read_json(str(pid_docid_map_file_path))
    assert doc_id_to_delete not in updated_pid_docid_map.values()

    if params_used.get("document_metadatas"):
        if docid_metadata_map_file_path.exists(): # File might be deleted if all entries removed
            updated_docid_metadata_map = srsly.read_json(str(docid_metadata_map_file_path))
            assert doc_id_to_delete not in updated_docid_metadata_map
            
    # Try to search for content of the deleted document
    results_after_delete = RAG.search(index_name=index_name, query=content_of_deleted_doc, k=5)
    found_deleted_id_in_results = any(res.get("document_id") == doc_id_to_delete for res in results_after_delete)
    assert not found_deleted_id_in_results, f"Document ID {doc_id_to_delete} was found in search results after deletion."