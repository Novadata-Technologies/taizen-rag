import pytest
import srsly
import os
from pathlib import Path

from ragatouille import RAGPretrainedModel
from ragatouille.utils import get_wikipedia_page

# Define a common index root for the test session using pytest's tmp_path_factory
@pytest.fixture(scope="session")
def session_index_root(tmp_path_factory):
    return tmp_path_factory.mktemp("ragatouille_e2e_tests_root")

@pytest.fixture(scope="session")
def model_name():
    return "colbert-ir/colbertv2.0" # Or a smaller test model if available

@pytest.fixture(scope="session")
def experiment_name():
    return "colbert" # Keep test experiments separate

@pytest.fixture(scope="session")
def miyazaki_index_name():
    return "Miyazaki"

@pytest.fixture(scope="session")
def toei_index_name():
    return "Toei"

@pytest.fixture(scope="session")
def miyazaki_index_path(session_index_root, experiment_name, miyazaki_index_name):
    return Path(session_index_root) / experiment_name / "indexes" / miyazaki_index_name

@pytest.fixture(scope="session")
def toei_index_path(session_index_root, experiment_name, toei_index_name):
    return Path(session_index_root) / experiment_name / "indexes" / toei_index_name

@pytest.fixture(scope="session")
def rag_model_for_indexing(model_name, session_index_root, experiment_name):
    """RAG model instance for indexing tests."""
    return RAGPretrainedModel.from_pretrained(
        model_name,
        index_root=str(session_index_root),
        experiment_name=experiment_name
    )

def test_multi_indexing(rag_model_for_indexing, miyazaki_index_name, miyazaki_index_path, toei_index_name, toei_index_path):
    RAG = rag_model_for_indexing
    with open("tests/data/miyazaki_wikipedia.txt", "r") as f:
        full_document = f.read()

    with open("tests/data/Toei_Animation_wikipedia.txt", "r") as f:
        full_document2 = f.read()

    # RAG.index now takes 'documents' and 'document_ids'
    # 'split_documents' is handled by the CorpusProcessor inside RAGPretrainedModel
    # 'max_document_length' is used as chunk_size for the default splitter
    RAG.index(
        index_name=miyazaki_index_name,
        documents=[full_document],
        document_ids=["miyazaki_doc_1"], # Must provide document_ids
        max_document_length=180,
    )
    RAG.index(
        index_name=toei_index_name,
        documents=[full_document2],
        document_ids=["toei_doc_1"], # Must provide document_ids
        max_document_length=180,
    )
    # ensure collections are stored to disk
    collection_path = miyazaki_index_path / "collection.json"
    collection_path2 = toei_index_path / "collection.json"
    assert collection_path.exists() and collection_path2.exists(), f"Collection file not found at {collection_path}"
    collection = srsly.read_json(str(collection_path))
    collection2 = srsly.read_json(str(collection_path2))
    assert len(collection) > 1, "Collection should have more than one chunk"
    assert len(collection2) > 1, "Collection should have more than one chunk"


def test_multi_index_search(model_name, session_index_root, experiment_name, miyazaki_index_name, miyazaki_index_path, toei_index_name, toei_index_path):
    # Ensure the index exists from test_indexing
    assert miyazaki_index_path.exists() and toei_index_path.exists(), f"Index paths {miyazaki_index_path} and {toei_index_path} must exist from a previous indexing step."

    # Use the new multi-index from_index method with index names
    RAG = RAGPretrainedModel.from_index(
        index_path_or_names=[miyazaki_index_name, toei_index_name],
        pretrained_model_name_or_path=model_name,
        index_root=str(session_index_root),
        experiment_name=experiment_name
    )
    k = 3
    # search now requires index_name
    results = RAG.search(index_name=miyazaki_index_name, query="What animation studio did Miyazaki found?", k=k)
    assert len(results) == k
    assert (
        "In April 1984, Miyazaki opened his own office in Suginami Ward"
        in results[0]["content"]
    )
    assert (
        "Hayao Miyazaki (宮崎 駿 or 宮﨑 駿, Miyazaki Hayao, [mijaꜜzaki hajao]; born January 5, 1941)"  # noqa
        in results[1]["content"]
    )
    assert (
        'Glen Keane said Miyazaki is a "huge influence" on Walt Disney Animation Studios and has been'  # noqa
        in results[2]["content"]
    )

    results = RAG.search(index_name=toei_index_name, query="When was Toei animation founded?", k=k)
    assert len(results) == k
    print("TOEI RESULTS", results)

    all_results = RAG.search(
        index_name=miyazaki_index_name,
        query=["What animation studio did Miyazaki found?", "Miyazaki son name"],
        k=k
    )
    assert (
        "In April 1984, Miyazaki opened his own office in Suginami Ward"
        in all_results[0][0]["content"]
    )
    assert (
        "Hayao Miyazaki (宮崎 駿 or 宮﨑 駿, Miyazaki Hayao, [mijaꜜzaki hajao]; born January 5, 1941)"  # noqa
        in all_results[0][1]["content"]
    )
    assert (
        'Glen Keane said Miyazaki is a "huge influence" on Walt Disney Animation Studios and has been'  # noqa
        in all_results[0][2]["content"]
    )
    assert (
        "== Early life ==\nHayao Miyazaki was born on January 5, 1941"
        in all_results[1][0]["content"]  # noqa
    )
    assert (
        "Directed by Isao Takahata, with whom Miyazaki would continue to collaborate for the remainder of his career"  # noqa
        in all_results[1][1]["content"]
    )
    actual = all_results[1][2]["content"]
    assert (
        "Specific works that have influenced Miyazaki include Animal Farm (1945)"
        in actual
        or "She met with Suzuki" in actual
    )
    print(all_results)


def test_multi_index_search_with_paths(model_name, miyazaki_index_path, toei_index_path):
    """Test loading multiple indexes using full paths instead of index names."""
    # Ensure the indexes exist from test_indexing
    assert miyazaki_index_path.exists() and toei_index_path.exists(), f"Index paths {miyazaki_index_path} and {toei_index_path} must exist from a previous indexing step."

    # Use the new multi-index from_index method with full paths
    RAG = RAGPretrainedModel.from_index(
        index_path_or_names=[str(miyazaki_index_path), str(toei_index_path)],
        pretrained_model_name_or_path=model_name
    )

    k = 3
    # Test searching both indexes
    miyazaki_results = RAG.search(index_name="Miyazaki", query="What animation studio did Miyazaki found?", k=k)
    assert len(miyazaki_results) == k
    assert "In April 1984, Miyazaki opened his own office in Suginami Ward" in miyazaki_results[0]["content"]

    toei_results = RAG.search(index_name="Toei", query="When was Toei animation founded?", k=k)
    assert len(toei_results) == k
    print("TOEI RESULTS (path-based):", toei_results)


@pytest.mark.skip(reason="experimental feature, needs careful review of add_to_index impact on existing data.")
def test_basic_CRUD_addition(model_name, miyazaki_index_name, miyazaki_index_path):
    assert miyazaki_index_path.exists(), f"Index path {miyazaki_index_path} must exist from a previous indexing step."
    collection_path = miyazaki_index_path / "collection.json"

    old_collection = srsly.read_json(str(collection_path))
    old_collection_len = len(old_collection)

    RAG = RAGPretrainedModel.from_index(
        index_path=str(miyazaki_index_path),
        pretrained_model_name_or_path=model_name
    )

    new_document_text = get_wikipedia_page("Studio_Ghibli")

    RAG.add_to_index(
        index_name=miyazaki_index_name,
        new_documents=[new_document_text],
        new_document_ids=["studio_ghibli_doc_CRUD"]
    )

    new_collection = srsly.read_json(str(collection_path))
    assert len(new_collection) > old_collection_len
    # Optional: Verify new content is searchable
    results = RAG.search(index_name=miyazaki_index_name, query="Tokuma Shoten", k=1)
    assert len(results) > 0
    assert "Tokuma Shoten" in results[0]["content"]
