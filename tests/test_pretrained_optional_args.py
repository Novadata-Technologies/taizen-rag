import os
import shutil

import pytest
import srsly

from ragatouille import RAGPretrainedModel

# Original global sample data
collection_samples = [
    "Hayao Miyazaki (宮崎 駿 or 宮﨑 駿, Miyazaki Hayao, [mijaꜜzaki hajao]; born January 5, 1941) is a Japanese animator, filmmaker, and manga artist. A co-founder of Studio Ghibli, he has attained international acclaim as a masterful storyteller and creator of Japanese animated feature films, and is widely regarded as one of the most accomplished filmmakers in the history of animation.\nBorn in Tokyo City in the Empire of Japan, Miyazaki expressed interest in manga and animation from an early age, and he joined Toei Animation in 1963. During his early years at Toei Animation he worked as an in-between artist and later collaborated with director Isao Takahata. Notable films to which Miyazaki contributed at Toei include Doggie March and Gulliver's Travels Beyond the Moon. He provided key animation to other films at Toei, such as Puss in Boots and Animal Treasure Island, before moving to A-Pro in 1971, where he co-directed Lupin the Third Part I alongside Takahata. After moving to Zuiyō Eizō (later known as Nippon Animation) in 1973, Miyazaki worked as an animator on World Masterpiece Theater, and directed the television series Future Boy Conan (1978). He joined Tokyo Movie Shinsha in 1979 to direct his first feature film The Castle of Cagliostro as well as the television series Sherlock Hound. In the same period, he also began writing and illustrating the manga Nausicaä of the Valley of the Wind (1982–1994), and he also directed the 1984 film adaptation produced by Topcraft.\nMiyazaki co-founded Studio Ghibli in 1985. He directed numerous films with Ghibli, including Laputa: Castle in the Sky (1986), My Neighbor Totoro (1988), Kiki's Delivery Service (1989), and Porco Rosso (1992). The films were met with critical and commercial success in Japan. Miyazaki's film Princess Mononoke was the first animated film ever to win the Japan Academy Prize for Picture of the Year, and briefly became the highest-grossing film in Japan following its release in 1997; its distribution to the Western world greatly increased Ghibli's popularity and influence outside Japan. His 2001 film Spirited Away became the highest-grossing film in Japanese history, winning the Academy Award for Best Animated Feature, and is frequently ranked among the greatest films of the 21st century. Miyazaki's later films—Howl's Moving Castle (2004), Ponyo (2008), and The Wind Rises (2013)—also enjoyed critical and commercial success.",
    "Studio Ghibli, Inc. (Japanese: 株式会社スタジオジブリ, Hepburn: Kabushiki gaisha Sutajio Jiburi) is a Japanese animation studio based in Koganei, Tokyo. It has a strong presence in the animation industry and has expanded its portfolio to include various media formats, such as short subjects, television commercials, and two television films. Their work has been well-received by audiences and recognized with numerous awards. Their mascot and most recognizable symbol, the character Totoro from the 1988 film My Neighbor Totoro, is a giant spirit inspired by raccoon dogs (tanuki) and cats (neko). Among the studio's highest-grossing films are Spirited Away (2001), Howl's Moving Castle (2004), and Ponyo (2008). Studio Ghibli was founded on June 15, 1985, by the directors Hayao Miyazaki and Isao Takahata and producer Toshio Suzuki, after acquiring Topcraft's assets. The studio has also collaborated with video game studios on the visual development of several games.Five of the studio's films are among the ten highest-grossing anime feature films made in Japan. Spirited Away is second, grossing 31.68 billion yen in Japan and over US$380 million worldwide, and Princess Mononoke is fourth, grossing 20.18 billion yen. Three of their films have won the Animage Grand Prix award, four have won the Japan Academy Prize for Animation of the Year, and five have received Academy Award nominations. Spirited Away won the 2002 Golden Bear and the 2003 Academy Award for Best Animated Feature.On August 3, 2014, Studio Ghibli temporarily suspended production following Miyazaki's retirement.",
]

document_ids_samples = ["miyazaki", "ghibli"]

document_metadatas_samples = [
    {"entity": "person", "source": "wikipedia"},
    {"entity": "organisation", "source": "wikipedia"},
]

PRETRAINED_MODEL_FOR_TESTS = "colbert-ir/colbertv2.0"
DEFAULT_EXPERIMENT_NAME = "colbert" # To match original path structure

@pytest.fixture(scope="session")
def persistent_temp_index_root(tmp_path_factory):
    path = tmp_path_factory.mktemp("temp_test_indexes_optional_args")
    yield str(path)
    # Cleanup is handled by pytest tmp_path_factory for session-scoped fixtures

@pytest.fixture(scope="session")
def RAG_from_pretrained_model_fixture(persistent_temp_index_root):
    # Pass experiment_name to maintain original path structure if tests relied on it
    return RAGPretrainedModel.from_pretrained(
        PRETRAINED_MODEL_FOR_TESTS,
        index_root=str(persistent_temp_index_root),
        experiment_name=DEFAULT_EXPERIMENT_NAME,
        verbose=0
    )

@pytest.fixture(scope="session")
def index_path_fixture_session(persistent_temp_index_root, index_creation_inputs_session):
    # Construct path as it was originally, assuming experiment_name="colbert"
    # index_root / experiment_name / "indexes" / index_name
    index_path = os.path.join(
        str(persistent_temp_index_root),
        DEFAULT_EXPERIMENT_NAME, # "colbert"
        "indexes",
        index_creation_inputs_session["index_name"],
    )
    return str(index_path)

@pytest.fixture(scope="session")
def collection_path_fixture_session(index_path_fixture_session):
    collection_path = os.path.join(index_path_fixture_session, "collection.json")
    return str(collection_path)

@pytest.fixture(scope="session")
def document_metadata_path_fixture_session(index_path_fixture_session):
    document_metadata_path = os.path.join(index_path_fixture_session, "docid_metadata_map.json")
    return str(document_metadata_path)

@pytest.fixture(scope="session")
def pid_docid_map_path_fixture_session(index_path_fixture_session):
    pid_docid_map_path = os.path.join(index_path_fixture_session, "pid_docid_map.json")
    return str(pid_docid_map_path)

# Renamed original fixture to avoid conflict if used directly by tests
# These are now session-scoped to match the dependent fixtures.
@pytest.fixture(
    scope="session",
    params=[
        {
            "collection": collection_samples, # Use original variable name for data
            "index_name": "no_optional_args",
            "split_documents": False,
        },
        {
            "collection": collection_samples,
            "document_ids": document_ids_samples, # Use original variable name
            "index_name": "with_docid",
            "split_documents": False,
        },
        {
            "collection": collection_samples,
            "document_metadatas": document_metadatas_samples, # Use original variable name
            "index_name": "with_metadata",
            "split_documents": False,
        },
        {
            "collection": collection_samples,
            "index_name": "with_split",
            "split_documents": True,
        },
        {
            "collection": collection_samples,
            "document_ids": document_ids_samples,
            "document_metadatas": document_metadatas_samples,
            "index_name": "with_docid_metadata",
            "split_documents": False,
        },
        {
            "collection": collection_samples,
            "document_ids": document_ids_samples,
            "index_name": "with_docid_split",
            "split_documents": True,
        },
        {
            "collection": collection_samples,
            "document_metadatas": document_metadatas_samples,
            "index_name": "with_metadata_split",
            "split_documents": True,
        },
        {
            "collection": collection_samples,
            "document_ids": document_ids_samples,
            "document_metadatas": document_metadatas_samples,
            "index_name": "with_docid_metadata_split",
            "split_documents": True,
        },
    ],
    ids=[
        "No optional arguments",
        "With document IDs",
        "With metadata",
        "With document splitting",
        "With document IDs and metadata",
        "With document IDs and splitting",
        "With metadata and splitting",
        "With document IDs, metadata, and splitting",
    ],
)
def index_creation_inputs_session(request):
    # This provides the raw parameters for each test case
    return request.param

@pytest.fixture(scope="session")
def create_index_session(RAG_from_pretrained_model_fixture, index_creation_inputs_session):
    api_call_params = index_creation_inputs_session.copy()

    if "collection" in api_call_params:
        api_call_params["documents"] = api_call_params.pop("collection")

    split_docs = api_call_params.pop("split_documents", False) # Default to False if not present
    if split_docs:
        api_call_params["max_document_length"] = 256 # Default for splitting
    else:
        api_call_params["max_document_length"] = 1_000_000 # Effectively no splitting

    # Ensure 'document_ids' and 'document_metadatas' are present if needed, or pass None
    api_call_params.setdefault("document_ids", None)
    api_call_params.setdefault("document_metadatas", None)

    # overwrite_index=True ensures a clean state for each parameterization if run independently
    # but since this is session scoped and parameterized, this might overwrite for subsequent params.
    # For strict adherence to "minimal change", we keep the original logic where overwrite
    # policy is implicitly handled by ColBERT (default "reuse").
    # If tests need pristine state per param set, they'd need function scope.
    # Given the original was session-scoped, let's assume sequential execution of params or reuse is fine.
    api_call_params["overwrite_index"] = True


    index_path = RAG_from_pretrained_model_fixture.index(**api_call_params)
    return index_path

# This fixture ensures that `index_creation_inputs_session` has `document_ids` populated
# for tests that rely on it later (like metadata checks).
@pytest.fixture(scope="session", autouse=True)
def populate_docids_in_inputs_session(
    create_index_session, # Ensures index is created first
    index_creation_inputs_session, # The specific parameters for current session iteration
    pid_docid_map_path_fixture_session, # Path to the pid_docid_map.json for this iteration
):
    # If document_ids were not initially provided for this parameter set
    if "document_ids" not in index_creation_inputs_session or index_creation_inputs_session["document_ids"] is None:
        # And if metadata was provided (implying IDs are important for mapping)
        # Or if we just want to ensure IDs are available for later tests like delete
        pid_docid_map_data = srsly.read_json(pid_docid_map_path_fixture_session)
        # Extract unique document IDs from the map
        seen_ids = set()
        # The PIDs are integers, values are the DocIDs.
        # RAGPretrainedModel ensures these are strings (UUIDs if auto-generated)
        effective_doc_ids = [
            str(x) # Ensure string type consistent with how RAGPretrainedModel handles it
            for x in list(pid_docid_map_data.values())
            if not (str(x) in seen_ids or seen_ids.add(str(x))) # Unique doc_ids
        ]
        index_creation_inputs_session["document_ids"] = effective_doc_ids
        # If metadatas were provided but IDs were not, this step is crucial
        # for aligning them for later assertions.
        # However, this might be tricky if the number of auto-generated IDs
        # doesn't match the number of metadatas provided initially.
        # The original test assumed this alignment; RAGPretrainedModel now requires
        # document_ids if document_metadatas is provided.
        # If document_ids was None but document_metadatas was not, an error should occur earlier.
        # This fixture primarily helps when both were None, or only collection was given.


# Tests now use the session-scoped fixtures.
# Each test function will run ONCE per parameter set defined in `index_creation_inputs_session`.
# The state of the index will persist across these test functions for a given parameter set.

def test_index_creation(create_index_session): # Uses the created index for the current param set
    assert os.path.exists(create_index_session), "Index path should exist."

def test_collection_creation(collection_path_fixture_session): # Path for current param set
    assert os.path.exists(collection_path_fixture_session)
    collection_data = srsly.read_json(collection_path_fixture_session)
    assert isinstance(collection_data, list)

def test_pid_docid_map_creation(pid_docid_map_path_fixture_session): # Path for current param set
    assert os.path.exists(pid_docid_map_path_fixture_session)
    pid_docid_map_data = srsly.read_json(pid_docid_map_path_fixture_session)
    assert isinstance(pid_docid_map_data, dict)

def test_document_metadata_creation(
    index_creation_inputs_session, document_metadata_path_fixture_session
):
    if "document_metadatas" in index_creation_inputs_session and index_creation_inputs_session["document_metadatas"] is not None:
        assert os.path.exists(document_metadata_path_fixture_session)
        document_metadata_dict = srsly.read_json(document_metadata_path_fixture_session)

        # Ensure document_ids were populated if metadatas are to be checked
        assert "document_ids" in index_creation_inputs_session and index_creation_inputs_session["document_ids"] is not None, \
            "document_ids must be available in inputs to check metadata mapping."

        assert set(document_metadata_dict.keys()) == set(index_creation_inputs_session["document_ids"])
        for doc_id, metadata in document_metadata_dict.items():
            idx = index_creation_inputs_session["document_ids"].index(doc_id)
            assert metadata == index_creation_inputs_session["document_metadatas"][idx]
    else:
        assert not os.path.exists(document_metadata_path_fixture_session)

def test_document_metadata_returned_in_search_results(
    RAG_from_pretrained_model_fixture, index_creation_inputs_session, index_path_fixture_session, persistent_temp_index_root
):
    # Use from_index to get a fresh RAG instance for searching this specific index parameterization
    # This ensures that the RAG instance is configured specifically for the index being tested.
    # The pretrained_model_name_or_path is crucial for the new API.
    RAG = RAGPretrainedModel.from_index(
        index_path=index_path_fixture_session,
        pretrained_model_name_or_path=PRETRAINED_MODEL_FOR_TESTS,
        index_root=str(persistent_temp_index_root),
        verbose=0
    )

    results = RAG.search(
        query="when was miyazaki born", # A generic query
        index_name=index_creation_inputs_session["index_name"]
    )

    if "document_metadatas" in index_creation_inputs_session and index_creation_inputs_session["document_metadatas"] is not None:
        assert "document_ids" in index_creation_inputs_session and index_creation_inputs_session["document_ids"] is not None, \
            "document_ids must be available in inputs to check metadata in search results."
        for result in results:
            assert "document_metadata" in result
            doc_id = result["document_id"]
            # This assumes doc_id from search result will be in the original list
            # which is true if IDs were provided or correctly inferred.
            if doc_id in index_creation_inputs_session["document_ids"]:
                 idx = index_creation_inputs_session["document_ids"].index(doc_id)
                 expected_metadata = index_creation_inputs_session["document_metadatas"][idx]
                 assert result["document_metadata"] == expected_metadata
    else:
        for result in results:
            assert "document_metadata" not in result


def test_add_to_existing_index(
    index_creation_inputs_session,
    index_path_fixture_session, # Path to the specific index for this param set
    pid_docid_map_path_fixture_session,
    document_metadata_path_fixture_session,
    persistent_temp_index_root
):
    # Instantiate RAG using from_index to ensure it's targeting the correct index path and root
    RAG = RAGPretrainedModel.from_index(
        index_path=index_path_fixture_session,
        pretrained_model_name_or_path=PRETRAINED_MODEL_FOR_TESTS,
        index_root=str(persistent_temp_index_root),
        verbose=0
    )

    # This relies on populate_docids_in_inputs_session to have run
    existing_doc_ids = index_creation_inputs_session.get("document_ids", [])

    new_doc_ids = ["mononoke_added", "sabaku_no_tami_added"]
    new_docs_to_add = [
        "Princess Mononoke is an epic film.",
        "People of the Desert is a manga by Miyazaki.",
    ]
    new_doc_metadata_to_add = [
        {"entity": "film", "source": "test_add"},
        {"entity": "manga", "source": "test_add"},
    ]

    RAG.add_to_index(
        index_name=index_creation_inputs_session["index_name"],
        new_documents=new_docs_to_add, # API change: new_collection -> new_documents
        new_document_ids=new_doc_ids,
        new_document_metadatas=new_doc_metadata_to_add,
    )

    pid_docid_map_data_after_add = srsly.read_json(pid_docid_map_path_fixture_session)
    doc_ids_in_map_after_add = set(list(pid_docid_map_data_after_add.values()))

    # Check for new docs
    for new_doc_id in new_doc_ids:
        assert new_doc_id in doc_ids_in_map_after_add

    # Check existing docs are still there
    for existing_doc_id in existing_doc_ids:
        assert existing_doc_id in doc_ids_in_map_after_add

    if os.path.exists(document_metadata_path_fixture_session):
        doc_metadata_dict_after_add = srsly.read_json(document_metadata_path_fixture_session)
        for new_doc_id in new_doc_ids:
            assert new_doc_id in doc_metadata_dict_after_add
        if "document_metadatas" in index_creation_inputs_session and index_creation_inputs_session["document_metadatas"] is not None:
            for existing_doc_id in existing_doc_ids:
                 assert existing_doc_id in doc_metadata_dict_after_add

def test_delete_from_index(
    index_creation_inputs_session,
    index_path_fixture_session, # Path to the specific index
    pid_docid_map_path_fixture_session,
    document_metadata_path_fixture_session,
    persistent_temp_index_root
):
    # Instantiate RAG using from_index to ensure it's targeting the correct index path and root
    RAG = RAGPretrainedModel.from_index(
        index_path=index_path_fixture_session,
        pretrained_model_name_or_path=PRETRAINED_MODEL_FOR_TESTS,
        index_root=str(persistent_temp_index_root),
        verbose=0
    )

    # This test assumes 'document_ids' were populated by populate_docids_in_inputs_session
    # or were part of the original index_creation_inputs_session.
    if not index_creation_inputs_session.get("document_ids"):
        pytest.skip("Cannot run delete test if no document_ids are defined for the index.")

    # To ensure this test doesn't interfere with subsequent parameterizations using the same files
    # if not re-instanced with from_index, we might need function-scoped RAG instances or careful state management.
    # However, the original structure was session-scoped. We proceed by trying to delete one of the *original* IDs.
    # This assumes the add_to_index test might have run before.

    doc_ids_before_delete_set = set(srsly.read_json(pid_docid_map_path_fixture_session).values())

    # Pick an ID to delete that was part of the original set for this param, if any
    # Or, if add_to_index ran, it could be one of those. Let's try to delete one of the "added" ones if present.
    id_to_delete = "mononoke_added" # From test_add_to_existing_index
    if id_to_delete not in doc_ids_before_delete_set:
        # If the "added" ID isn't there (e.g. add test skipped or failed), try an original one
        if index_creation_inputs_session["document_ids"]:
            id_to_delete = index_creation_inputs_session["document_ids"][0]
        else:
            pytest.skip("No clear ID to delete for this test run.")


    RAG.delete_from_index(
        index_name=index_creation_inputs_session["index_name"],
        document_ids=[id_to_delete], # API change: expects List[str]
    )

    pid_docid_map_data_after_delete = srsly.read_json(pid_docid_map_path_fixture_session)
    doc_ids_in_map_after_delete = set(list(pid_docid_map_data_after_delete.values()))

    assert id_to_delete not in doc_ids_in_map_after_delete

    if "document_metadatas" in index_creation_inputs_session and index_creation_inputs_session["document_metadatas"] is not None:
        if os.path.exists(document_metadata_path_fixture_session): # Check if metadata file still exists
            doc_metadata_dict_after_delete = srsly.read_json(document_metadata_path_fixture_session)
            assert id_to_delete not in doc_metadata_dict_after_delete
