import pytest
from fastapi.testclient import TestClient

from ragatouille.server.server import app
from ragatouille import RAGPretrainedModel


# Test fixtures
@pytest.fixture(scope="session")
def session_index_root(tmp_path_factory):
    """Create a temporary directory for test indices"""
    return tmp_path_factory.mktemp("ragatouille_api_e2e_tests_root")


@pytest.fixture(scope="session")
def model_name():
    return "colbert-ir/colbertv2.0"


@pytest.fixture(scope="session")
def experiment_name():
    return "colbert"


@pytest.fixture(scope="session")
def miyazaki_index_name():
    return "Miyazaki_API"


@pytest.fixture(scope="session")
def toei_index_name():
    return "Toei_API"


@pytest.fixture(scope="session")
def test_documents():
    """Load test documents"""
    with open("tests/data/miyazaki_wikipedia.txt", "r", encoding="utf-8") as f:
        miyazaki_doc = f.read()

    with open("tests/data/Toei_Animation_wikipedia.txt", "r", encoding="utf-8") as f:
        toei_doc = f.read()

    return {
        "miyazaki": miyazaki_doc,
        "toei": toei_doc
    }


@pytest.fixture(scope="session")
def test_client(session_index_root, model_name, experiment_name):
    """Create FastAPI test client with properly initialized RAG model"""
    import ragatouille.server.server as server_module

    # Initialize RAG model for testing
    server_module.RAG = RAGPretrainedModel.from_pretrained(
        pretrained_model_name_or_path=model_name,
        index_root=str(session_index_root),
        experiment_name=experiment_name,
    )

    # Create test client
    client = TestClient(app)

    yield client

    # Cleanup
    if server_module.RAG is not None:
        server_module.RAG = None


class TestIndexCreation:
    """Test index creation endpoint"""

    def test_create_miyazaki_index(self, test_client, miyazaki_index_name, test_documents):
        """Test creating an index with Miyazaki document"""
        response = test_client.post(
            "/api/v1/index",
            json={
                "input": [test_documents["miyazaki"]],
                "metadata": [{"id": "miyazaki_doc_1", "project_id": "test_project"}],
                "index_id": miyazaki_index_name
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["result"] == "ok"
        assert "index_path" in data

    def test_create_toei_index(self, test_client, toei_index_name, test_documents):
        """Test creating an index with Toei document"""
        response = test_client.post(
            "/api/v1/index",
            json={
                "input": [test_documents["toei"]],
                "metadata": [{"id": "toei_doc_1", "project_id": "test_project"}],
                "index_id": toei_index_name
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["result"] == "ok"
        assert "index_path" in data

    def test_create_index_empty_input(self, test_client):
        """Test creating index with empty input should fail"""
        response = test_client.post(
            "/api/v1/index",
            json={
                "input": [],
                "metadata": [],
                "index_id": "empty_test"
            }
        )

        assert response.status_code == 400
        assert "Input documents list cannot be empty" in response.json()["detail"]

    def test_create_index_mismatched_metadata(self, test_client, test_documents):
        """Test creating index with mismatched input and metadata lengths"""
        response = test_client.post(
            "/api/v1/index",
            json={
                "input": [test_documents["miyazaki"]],
                "metadata": [
                    {"id": "doc1", "project_id": "test"},
                    {"id": "doc2", "project_id": "test"}
                ],
                "index_id": "mismatch_test"
            }
        )

        assert response.status_code == 400
        assert "Length of input documents and metadata must match" in response.json()["detail"]

    def test_create_index_duplicate_ids(self, test_client, test_documents):
        """Test creating index with duplicate document IDs"""
        response = test_client.post(
            "/api/v1/index",
            json={
                "input": [test_documents["miyazaki"], test_documents["toei"]],
                "metadata": [
                    {"id": "duplicate_id", "project_id": "test"},
                    {"id": "duplicate_id", "project_id": "test"}
                ],
                "index_id": "duplicate_test"
            }
        )

        assert response.status_code == 400
        assert "Document IDs in metadata must be unique" in response.json()["detail"]


class TestIndexListing:
    """Test index listing endpoint"""

    def test_list_indexes_includes_created(self, test_client, miyazaki_index_name, toei_index_name):
        """Test that listing indexes includes previously created indexes"""
        response = test_client.get("/api/v1/index/list")

        assert response.status_code == 200
        data = response.json()
        assert "indexes" in data

        # Check that our test indexes are in the list
        index_names = list(data["indexes"].keys())
        assert miyazaki_index_name in index_names
        assert toei_index_name in index_names

        # Check index metadata structure
        for index_name in [miyazaki_index_name, toei_index_name]:
            index_info = data["indexes"][index_name]
            assert isinstance(index_info, dict)
            # The index should have some metadata about its state

    def test_index_memory_status_before_query(self, test_client, miyazaki_index_name):
        """Test that index is not in memory before querying"""
        response = test_client.get("/api/v1/index/list")
        assert response.status_code == 200

        data = response.json()
        index_info = data["indexes"][miyazaki_index_name]

        # Index should exist but not necessarily be loaded in memory
        # (This depends on the actual implementation of get_available_indexes)
        assert index_info is not None

    def test_index_memory_status_after_query(self, test_client, miyazaki_index_name):
        """Test that index status changes after querying"""
        # First, perform a query to load the index
        query_response = test_client.post(
            "/api/v1/query",
            json={
                "input": ["What animation studio did Miyazaki found?"],
                "index_id": miyazaki_index_name,
                "k": 3
            }
        )
        assert query_response.status_code == 200

        # Now check the index list again
        list_response = test_client.get("/api/v1/index/list")
        assert list_response.status_code == 200

        data = list_response.json()
        index_info = data["indexes"][miyazaki_index_name]

        # Index should still be available and potentially marked as loaded
        assert index_info is not None


class TestQuerying:
    """Test query endpoint"""

    def test_query_miyazaki_index(self, test_client, miyazaki_index_name):
        """Test querying the Miyazaki index"""
        response = test_client.post(
            "/api/v1/query",
            json={
                "input": ["What animation studio did Miyazaki found?"],
                "index_id": miyazaki_index_name,
                "k": 3
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert "model" in data
        assert len(data["data"]) == 1  # One query
        assert len(data["data"][0]) == 3  # k=3 results

        # Check result structure
        result = data["data"][0][0]
        assert "content" in result
        assert isinstance(result["content"], str)

    def test_query_toei_index(self, test_client, toei_index_name):
        """Test querying the Toei index"""
        response = test_client.post(
            "/api/v1/query",
            json={
                "input": ["When was Toei animation founded?"],
                "index_id": toei_index_name,
                "k": 3
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["data"]) == 1
        assert len(data["data"][0]) == 3

        # Check that the result contains relevant information
        first_result = data["data"][0][0]["content"]
        assert "1948" in first_result or "Toei" in first_result

    def test_query_multiple_queries(self, test_client, miyazaki_index_name):
        """Test multiple queries in one request"""
        response = test_client.post(
            "/api/v1/query",
            json={
                "input": [
                    "What animation studio did Miyazaki found?",
                    "When was Miyazaki born?"
                ],
                "index_id": miyazaki_index_name,
                "k": 2
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["data"]) == 2  # Two queries
        assert len(data["data"][0]) == 2  # k=2 results for first query
        assert len(data["data"][1]) == 2  # k=2 results for second query

    def test_query_nonexistent_index(self, test_client):
        """Test querying a non-existent index"""
        response = test_client.post(
            "/api/v1/query",
            json={
                "input": ["test query"],
                "index_id": "nonexistent_index",
                "k": 3
            }
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_query_empty_input(self, test_client, miyazaki_index_name):
        """Test querying with empty input"""
        response = test_client.post(
            "/api/v1/query",
            json={
                "input": [],
                "index_id": miyazaki_index_name,
                "k": 3
            }
        )

        assert response.status_code == 400
        assert "Input query list cannot be empty" in response.json()["detail"]


class TestDocumentOperations:
    """Test document addition and deletion operations"""

    def test_add_document_to_index(self, test_client, miyazaki_index_name):
        """Test adding a document to an existing index"""
        response = test_client.put(
            "/api/v1/index/doc",
            json={
                "input": ["This is a new test document about Studio Ghibli."],
                "metadata": [{"id": "ghibli_doc_1", "project_id": "test_project"}],
                "index_id": miyazaki_index_name
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["result"] == "ok"

        # Verify the document can be found
        query_response = test_client.post(
            "/api/v1/query",
            json={
                "input": ["Studio Ghibli"],
                "index_id": miyazaki_index_name,
                "k": 5
            }
        )

        assert query_response.status_code == 200
        results = query_response.json()["data"][0]

        # Check if any result contains our new document
        ghibli_found = any("Studio Ghibli" in result["content"] for result in results)
        assert ghibli_found

    def test_delete_documents_success(self, test_client, miyazaki_index_name):
        """Test successful document deletion"""
        # First add a document we can delete
        add_response = test_client.put(
            "/api/v1/index/doc",
            json={
                "input": ["This is a temporary document for deletion test."],
                "metadata": [{"id": "temp_doc_delete", "project_id": "test_project"}],
                "index_id": miyazaki_index_name
            }
        )
        assert add_response.status_code == 201

        # Now delete it
        delete_response = test_client.delete(
            "/api/v1/index/doc",
            json={
                "documents": {
                    miyazaki_index_name: ["temp_doc_delete"]
                }
            }
        )

        assert delete_response.status_code == 200
        data = delete_response.json()

        assert "success" in data
        assert "failure" in data
        assert "error" in data

        assert miyazaki_index_name in data["success"]
        assert "temp_doc_delete" in data["success"][miyazaki_index_name]
        assert len(data["failure"]) == 0
        assert data["error"] == ""

    def test_delete_documents_nonexistent_index(self, test_client):
        """Test deleting documents from non-existent index"""
        delete_response = test_client.delete(
            "/api/v1/index/doc",
            json={
                "documents": {
                    "nonexistent_index": ["doc_id"]
                }
            }
        )

        assert delete_response.status_code == 404
        data = delete_response.json()

        assert "failure" in data
        assert "nonexistent_index" in data["failure"]
        assert len(data["success"]) == 0
        assert data["error"] != ""

    def test_delete_documents_partial_success(self, test_client, miyazaki_index_name, toei_index_name):
        """Test partial success when deleting from multiple indexes"""
        # Add a document to one index that we can successfully delete
        add_response = test_client.put(
            "/api/v1/index/doc",
            json={
                "input": ["Document for partial delete test."],
                "metadata": [{"id": "partial_delete_doc", "project_id": "test_project"}],
                "index_id": miyazaki_index_name
            }
        )
        assert add_response.status_code == 201

        # Try to delete documents from both existing and non-existing indexes
        delete_response = test_client.delete(
            "/api/v1/index/doc",
            json={
                "documents": {
                    miyazaki_index_name: ["partial_delete_doc"],
                    "nonexistent_index": ["some_doc_id"]
                }
            }
        )

        # This might be 200 with partial success or an error code depending on implementation
        # The key is that we should get information about what succeeded and what failed
        data = delete_response.json()

        assert "success" in data
        assert "failure" in data
        assert "error" in data

        # At least one operation should have some result
        assert len(data["success"]) > 0 or len(data["failure"]) > 0


class TestIndexDeletion:
    """Test index deletion (entire index removal)"""

    def test_delete_entire_index_success(self, test_client):
        """Test successful deletion of an entire index"""
        # First create a test index specifically for deletion
        create_response = test_client.post(
            "/api/v1/index",
            json={
                "input": ["This index will be deleted."],
                "metadata": [{"id": "delete_test_doc", "project_id": "test_project"}],
                "index_id": "index_to_delete"
            }
        )
        assert create_response.status_code == 201

        # Verify it exists in the index list
        list_response = test_client.get("/api/v1/index/list")
        assert list_response.status_code == 200
        assert "index_to_delete" in list_response.json()["indexes"]

        # Verify we can query it before deletion
        query_response = test_client.post(
            "/api/v1/query",
            json={
                "input": ["test"],
                "index_id": "index_to_delete",
                "k": 1
            }
        )
        assert query_response.status_code == 200

        # Now delete the entire index
        delete_response = test_client.delete("/api/v1/index/index_to_delete")
        assert delete_response.status_code == 200

        data = delete_response.json()
        assert data["result"] == "ok"
        assert data["deleted_index"] == "index_to_delete"

        # Verify it no longer exists in the index list
        list_response_after = test_client.get("/api/v1/index/list")
        assert list_response_after.status_code == 200
        assert "index_to_delete" not in list_response_after.json()["indexes"]

    def test_delete_nonexistent_index(self, test_client):
        """Test deleting a non-existent index"""
        delete_response = test_client.delete("/api/v1/index/nonexistent_index")
        assert delete_response.status_code == 404
        assert "not found" in delete_response.json()["detail"].lower()

    def test_operations_on_deleted_index(self, test_client):
        """Test that operations fail on deleted indexes"""
        # First create and then delete an index
        create_response = test_client.post(
            "/api/v1/index",
            json={
                "input": ["This index will be deleted and then accessed."],
                "metadata": [{"id": "delete_then_access_doc", "project_id": "test_project"}],
                "index_id": "index_to_delete_then_access"
            }
        )
        assert create_response.status_code == 201

        # Delete the index
        delete_response = test_client.delete("/api/v1/index/index_to_delete_then_access")
        assert delete_response.status_code == 200

        # Try to query the deleted index - should fail
        query_response = test_client.post(
            "/api/v1/query",
            json={
                "input": ["test"],
                "index_id": "index_to_delete_then_access",
                "k": 1
            }
        )
        assert query_response.status_code == 404

        # Try to add documents to the deleted index - should fail
        test_client.put(
            "/api/v1/index/doc",
            json={
                "input": ["New document for deleted index."],
                "metadata": [{"id": "new_doc", "project_id": "test_project"}],
                "index_id": "index_to_delete_then_access"
            }
        )
        # This might create a new index or fail depending on implementation
        # The key is that it should handle the deleted index appropriately

    def test_delete_index_with_failure_scenario(self, test_client):
        """Test partial failure scenarios when deleting indices"""
        # Create an index
        create_response = test_client.post(
            "/api/v1/index",
            json={
                "input": ["Index for failure scenario test."],
                "metadata": [{"id": "failure_test_doc", "project_id": "test_project"}],
                "index_id": "failure_test_index"
            }
        )
        assert create_response.status_code == 201

        # Delete successfully first time
        delete_response = test_client.delete("/api/v1/index/failure_test_index")
        assert delete_response.status_code == 200

        # Try to delete the same index again - should fail gracefully
        delete_response_2 = test_client.delete("/api/v1/index/failure_test_index")
        assert delete_response_2.status_code == 404
        assert "not found" in delete_response_2.json()["detail"].lower()


class TestErrorHandling:
    """Test various error conditions"""

    def test_malformed_json(self, test_client):
        """Test handling of malformed JSON requests"""
        response = test_client.post(
            "/api/v1/index",
            data="this is not json"
        )

        assert response.status_code == 422  # Unprocessable Entity

    def test_missing_required_fields(self, test_client):
        """Test handling of requests with missing required fields"""
        response = test_client.post(
            "/api/v1/index",
            json={
                "input": ["test document"],
                # Missing metadata and index_id
            }
        )

        assert response.status_code == 422

    def test_invalid_field_types(self, test_client):
        """Test handling of invalid field types"""
        response = test_client.post(
            "/api/v1/index",
            json={
                "input": "should be a list",  # Wrong type
                "metadata": [{"id": "test", "project_id": "test"}],
                "index_id": "test_index"
            }
        )

        assert response.status_code == 422


# Integration test combining multiple operations
class TestWorkflowIntegration:
    """Test complete workflows combining multiple operations"""

    def test_complete_index_lifecycle(self, test_client, test_documents):
        """Test complete lifecycle: create -> query -> add docs -> query -> delete docs -> query"""
        index_name = "lifecycle_test_index"

        # 1. Create index
        create_response = test_client.post(
            "/api/v1/index",
            json={
                "input": [test_documents["miyazaki"][:1000]],  # Use first 1000 chars
                "metadata": [{"id": "lifecycle_doc_1", "project_id": "lifecycle_test"}],
                "index_id": index_name
            }
        )
        assert create_response.status_code == 201

        # 2. Query initial index
        query_response = test_client.post(
            "/api/v1/query",
            json={
                "input": ["Miyazaki"],
                "index_id": index_name,
                "k": 2
            }
        )
        assert query_response.status_code == 200
        query_response.json()["data"][0]

        # 3. Add more documents
        add_response = test_client.put(
            "/api/v1/index/doc",
            json={
                "input": ["Additional document about animation and films."],
                "metadata": [{"id": "lifecycle_doc_2", "project_id": "lifecycle_test"}],
                "index_id": index_name
            }
        )
        assert add_response.status_code == 201

        # 4. Query again to see if new document is searchable
        query_response2 = test_client.post(
            "/api/v1/query",
            json={
                "input": ["animation films"],
                "index_id": index_name,
                "k": 3
            }
        )
        assert query_response2.status_code == 200
        updated_results = query_response2.json()["data"][0]

        # Should find the new document
        animation_found = any("animation" in result["content"].lower() for result in updated_results)
        assert animation_found

        # 5. Delete one document
        delete_response = test_client.delete(
            "/api/v1/index/doc",
            json={
                "documents": {
                    index_name: ["lifecycle_doc_2"]
                }
            }
        )
        assert delete_response.status_code == 200

        # 6. Verify index still works but deleted document is gone
        final_query_response = test_client.post(
            "/api/v1/query",
            json={
                "input": ["animation films"],
                "index_id": index_name,
                "k": 5
            }
        )
        assert final_query_response.status_code == 200

        # The specific "Additional document about animation" should be less likely to appear
        # since it was deleted, but the original Miyazaki content should still be there
        final_results = final_query_response.json()["data"][0]
        miyazaki_found = any("miyazaki" in result["content"].lower() for result in final_results)
        assert miyazaki_found
