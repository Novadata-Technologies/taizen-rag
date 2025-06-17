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
def miyazaki_index_path(session_index_root, experiment_name, miyazaki_index_name):
    return Path(session_index_root) / experiment_name / "indexes" / miyazaki_index_name

@pytest.fixture(scope="session")
def rag_model_for_indexing(model_name, session_index_root, experiment_name):
    """RAG model instance for indexing tests."""
    return RAGPretrainedModel.from_pretrained(
        model_name,
        index_root=str(session_index_root),
        experiment_name=experiment_name
    )

def test_indexing(rag_model_for_indexing, miyazaki_index_name, miyazaki_index_path):
    RAG = rag_model_for_indexing
    with open("tests/data/miyazaki_wikipedia.txt", "r") as f:
        full_document = f.read()

    # RAG.index now takes 'documents' and 'document_ids'
    # 'split_documents' is handled by the CorpusProcessor inside RAGPretrainedModel
    # 'max_document_length' is used as chunk_size for the default splitter
    RAG.index(
        index_name=miyazaki_index_name,
        documents=[full_document],
        document_ids=["miyazaki_doc_1"], # Must provide document_ids
        max_document_length=180,
        split_documents=True
    )
    # ensure collection is stored to disk
    collection_path = miyazaki_index_path / "collection.json"
    assert collection_path.exists(), f"Collection file not found at {collection_path}"
    collection = srsly.read_json(str(collection_path))
    assert len(collection) > 1, "Collection should have more than one chunk"


def test_search(model_name, miyazaki_index_name, miyazaki_index_path):
    # Ensure the index exists from test_indexing
    assert miyazaki_index_path.exists(), f"Index path {miyazaki_index_path} must exist from a previous indexing step."

    # from_index now also requires pretrained_model_name_or_path
    RAG = RAGPretrainedModel.from_index(
        index_path=str(miyazaki_index_path),
        pretrained_model_name_or_path=model_name
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
