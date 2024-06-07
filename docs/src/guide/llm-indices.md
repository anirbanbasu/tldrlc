# Language model providers, documents, index and graph storage

## Supported language model providers

You can specify the language model provider to use by using the environment variable `LLM_PROVIDER`, which defaults to `Ollama`, if not specified. The supported language model providers are:

1. Cohere.
2. Ollama.
3. Open AI.

If using [Ollama](https://ollama.com/), you will also need to install it or, point the chatbot to a remotely hosted Ollama server. You also need to pull the Ollama model that you specify with `OLLAMA_MODEL` environment variable using `ollama pull <model-name>` (replace `<model-name>` with the actual model that you want to use) on your Ollama server. Check the [available Ollama models](https://ollama.com/library).

Open AI can be used by specifying an `OPENAI_API_KEY`, an `OPENAI_MODEL`, and by choosing `Open AI` as the `LLM_PROVIDER`. Follow [this link](https://platform.openai.com/account/api-keys) to get an Open AI API key. Similarly, Cohere can be used by specifying a `COHERE_API_KEY`, a `COHERE_MODEL` (which defaults to `command-r-plus`), and by choosing `Cohere` as the `LLM_PROVIDER`. Follow [this link](https://cohere.com/pricing) to obtain a Cohere API key.

See the settings in the `.env.template` file customisation of the LLM settings.

## Documents and index storage

A [Redis stack installation](https://redis.io/docs/install/install-stack/) is required to store the parsed document chunks and the index, which can be reloaded without having to rebuild the index from the sources. Although the default settings are okay, refer to the [Redis persistence configuration](https://redis.io/docs/management/persistence/), if necessary. Alternative to a local Redis installation, you can also opt for a remote Redis installation such as one on [Render](https://render.com/). Depending on your hosting plan, various limits will apply on such cloud-based managed Redis installations. 

## Graph database

Similarly, you will need [Neo4j graph database](https://neo4j.com/). You can either install it locally or use a the [Neo4j Aura DB](https://neo4j.com/cloud/platform/aura-graph-database/), which is a cloud-hosted version of Neo4j.

_Note that unless you specify `NEO4J_DISABLE = "True"` to disable Neo4J and use an in-memory graph database, the Neo4J server must be accessible using the specified connection credentials. Otherwise, the application will display an error at the time of starting up._