import pytest
import shutil
from pathlib import Path
from ragatouille import RAGPretrainedModel

# Constants for tests
PRETRAINED_MODEL_FOR_LOADING_TESTS = "colbert-ir/colbertv2.0" # Using a real, small model
# Alternative for faster tests if available: "hf-internal-testing/tiny-bert-colbert" - but this might not work with ColBERT logic directly
TEST_EXPERIMENT_LOADING = "ragatouille_loading_tests"

@pytest.fixture(scope="session")
def loading_test_index_root(tmp_path_factory):
    """Creates a session-wide temporary root directory for indices created during loading tests."""
    path = tmp_path_factory.mktemp("loading_tests_indices")
    yield str(path)
    shutil.rmtree(path, ignore_errors=True) # Clean up after all tests in session

def test_from_pretrained_loads_model(loading_test_index_root):
    """Tests if RAGPretrainedModel.from_pretrained successfully loads the underlying ColBERT model."""
    rag_instance = RAGPretrainedModel.from_pretrained(
        pretrained_model_name_or_path=PRETRAINED_MODEL_FOR_LOADING_TESTS,
        index_root=loading_test_index_root,
        experiment_name=TEST_EXPERIMENT_LOADING,
        verbose=0
    )
    assert isinstance(rag_instance, RAGPretrainedModel), "Should return a RAGPretrainedModel instance."
    assert rag_instance.model is not None, "Internal ColBERT model should be initialized."
    assert hasattr(rag_instance.model, 'inference_ckpt'), "ColBERT model should have an inference_ckpt."
    assert rag_instance.model.inference_ckpt is not None, "inference_ckpt should be loaded."
    assert rag_instance.model.base_pretrained_model_name_or_path == PRETRAINED_MODEL_FOR_LOADING_TESTS

def test_from_index_loads_model_and_specific_index(loading_test_index_root):
    """
    Tests if RAGPretrainedModel.from_index can load a model and correctly
    initialize with the specified index.
    """
    index_name_for_test = "my_loading_test_index"
    documents_for_index = ["This is a test document for loading from index."]
    doc_ids_for_index = ["test_doc_loader_01"]

    # Step 1: Create an index to load from
    rag_for_creation = RAGPretrainedModel.from_pretrained(
        pretrained_model_name_or_path=PRETRAINED_MODEL_FOR_LOADING_TESTS,
        index_root=loading_test_index_root,
        experiment_name=TEST_EXPERIMENT_LOADING,
        verbose=0
    )
    created_index_path_str = rag_for_creation.index(
        index_name=index_name_for_test,
        documents=documents_for_index,
        document_ids=doc_ids_for_index,
        overwrite_index=True
    )
    created_index_path = Path(created_index_path_str)
    assert created_index_path.exists(), "Index directory for loading test was not created."

    # Step 2: Load from the created index
    rag_loaded_from_index = RAGPretrainedModel.from_index(
        index_path=str(created_index_path),
        pretrained_model_name_or_path=PRETRAINED_MODEL_FOR_LOADING_TESTS, # Crucial for new API
        verbose=0
    )

    assert isinstance(rag_loaded_from_index, RAGPretrainedModel), "from_index should return RAGPretrainedModel."
    assert rag_loaded_from_index.model is not None, "Internal ColBERT model should be initialized via from_index."
    assert hasattr(rag_loaded_from_index.model, 'inference_ckpt'), "ColBERT model (from_index) should have inference_ckpt."
    assert rag_loaded_from_index.model.inference_ckpt is not None, "inference_ckpt (from_index) should be loaded."

    # Check if the ColBERT model correctly identifies the initially loaded index
    # The 'initial_index_name' is set in ColBERT.__init__ if an index is loaded at startup.
    # from_index in RAGPretrainedModel passes the inferred index name as 'initial_index_name'
    # to from_pretrained, which then passes it to ColBERT.
    assert rag_loaded_from_index.model.index_configs.get(index_name_for_test) is not None, \
        f"Index '{index_name_for_test}' config not found in loaded model."
    
    # Verify the model's index_root and experiment_name are correctly set through from_index logic
    expected_index_root = str(created_index_path.parent.parent.parent) # index_root/experiment/indexes/index_name
    assert rag_loaded_from_index.model.index_root == expected_index_root
    assert rag_loaded_from_index.model.experiment_name == TEST_EXPERIMENT_LOADING


    # Step 3: Perform a simple search to ensure the loaded index is active and usable
    search_query = "test document"
    results = rag_loaded_from_index.search(
        index_name=index_name_for_test,
        query=search_query,
        k=1
    )
    assert isinstance(results, list), "Search should return a list."
    if documents_for_index: # If we actually indexed something
        assert len(results) > 0, "Search returned no results on a supposedly loaded index."
        assert results[0]["document_id"] == doc_ids_for_index[0], "Search did not retrieve the correct document."

@pytest.mark.skip(reason="Test intent unclear or needs refactoring for current API. Covered by other e2e search tests.")
def test_searcher():
    """
    This test was originally skipped and its intent regarding 'searcher' properties
    is not directly testable via RAGPretrainedModel's public API in a meaningful way
    beyond what `test_search` in e2e tests or `test_from_index_loads_model_and_index` cover.
    The internal `Searcher` object management is an implementation detail of `ColBERT` and `ModelIndex`.
    """
    pass