# Language model providers

## Supported language model providers

You can specify the language model provider to use by using the environment variable `LLM_PROVIDER`, which defaults to `Ollama`, if not specified. The supported language model providers are:

1. Cohere.
2. Ollama.
3. Open AI.

If using [Ollama](https://ollama.com/), you will also need to install it or, point the chatbot to a remotely hosted Ollama server. You also need to pull the Ollama model that you specify with `OLLAMA_MODEL` environment variable using `ollama pull <model-name>` (replace `<model-name>` with the actual model that you want to use) on your Ollama server. Check the [available Ollama models](https://ollama.com/library).

Open AI can be used by specifying an `OPENAI_API_KEY`, an `OPENAI_MODEL`, and by choosing `Open AI` as the `LLM_PROVIDER`. Follow [this link](https://platform.openai.com/account/api-keys) to get an Open AI API key. Similarly, Cohere can be used by specifying a `COHERE_API_KEY`, a `COHERE_MODEL` (which defaults to `command-r-plus`), and by choosing `Cohere` as the `LLM_PROVIDER`. Follow [this link](https://cohere.com/pricing) to obtain a Cohere API key.

See the settings in the `.env.template` file customisation of the LLM settings.