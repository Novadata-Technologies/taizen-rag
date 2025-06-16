This is a fork of RAGatouille, a wrapper over Colbert-AI which provides a straightforward API.

Right now to adapt it to Taizen use we need:
 1. We need an index by project_id. Right now each index is associated to a model instance, so this can clutter memory pretty quick. We need to change something so the model is loaded 1 time and each index is stored in memory.
 2. For the full workflow: The API workers would need to spin up a task. With only 1 worker we would then perform the search or index. (Maybe, since the worker is spinned up with parameters we can pass that to the worker and let each worker instantiate the index itself)
 3. We can wrap up the endpoints with batched for a better use of parallel resources. We have to be careful because requests would have to be split by project_id...
 4. Right now each index has to be manually created, we should provide an endpoint for that.
