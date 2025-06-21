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


def test_multi_index_search(model_name, miyazaki_index_name, miyazaki_index_path, toei_index_name, toei_index_path):
    # Ensure the index exists from test_indexing
    assert miyazaki_index_path.exists() and toei_index_path.exists(), f"Index paths {miyazaki_index_path} and {toei_index_path} must exist from a previous indexing step."

    # Load from one index, but model will have access to all indexes in the same experiment
    RAG = RAGPretrainedModel.from_index(
        index_path=str(miyazaki_index_path),
        pretrained_model_name_or_path=model_name
    )
    k = 3
    # search now requires index_name
    results = RAG.search(index_name=miyazaki_index_name, query="What animation studio did Miyazaki found?", k=k)
    assert len(results) == k
    assert any('1984' in passage['content'] for passage in results)


    results = RAG.search(index_name=toei_index_name, query="When was Toei animation founded?", k=k)
    assert len(results) == k
    assert any('1948' in passage['content'] for passage in results)

    all_results = RAG.search(
        index_name=miyazaki_index_name,
        query=["What animation studio did Miyazaki found?", "Miyazaki son name"],
        k=k
    )
    assert any('1984' in passage['content'] for passage in all_results[0])
    assert any('keisuke' in passage['content'].lower() or 'goro' in passage['content'].lower() for passage in all_results[1])

    all_results = RAG.search(
        index_name=toei_index_name,
        query=["When was Toei animation founded?", "biggest Toei hits"],
        k=k
    )
    assert any('1948' in passage['content'] for passage in all_results[0])
    assert any('Dr. Slump' in passage['content'] for passage in all_results[1])
    assert any('Dragon Ball' in passage['content'] for passage in all_results[1])
    assert any('One Piece' in passage['content'] for passage in all_results[1])


def test_multi_index_search_alternative_loading(model_name, miyazaki_index_path, toei_index_path):
    """Test that loading from the other index also works."""
    # Ensure the indexes exist from test_indexing
    assert miyazaki_index_path.exists() and toei_index_path.exists(), f"Index paths {miyazaki_index_path} and {toei_index_path} must exist from a previous indexing step."

    # Load from the Toei index this time
    RAG = RAGPretrainedModel.from_index(
        index_path=str(toei_index_path),
        pretrained_model_name_or_path=model_name
    )

    k = 3
    # Test searching both indexes (should work because they're in the same experiment)
    miyazaki_results = RAG.search(index_name="Miyazaki", query="What animation studio did Miyazaki found?", k=k)
    assert len(miyazaki_results) == k
    assert "In April 1984, Miyazaki opened his own office in Suginami Ward" in miyazaki_results[0]["content"]

    toei_results = RAG.search(index_name="Toei", query="When was Toei animation founded?", k=k)
    assert len(toei_results) == k


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
