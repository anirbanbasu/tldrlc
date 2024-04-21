[![Python ruff linter format-checker](https://github.com/anirbanbasu/tldrlc/actions/workflows/python-app.yml/badge.svg)](https://github.com/anirbanbasu/tldrlc/actions/workflows/python-app.yml)

# Too Long, Didn't Read, Let's Chat (TLDRLC)

TLDRLC is an experimental chatbot prototype that allows users to chat with various data sources. This chatbot utilises a _knowledge graph_ based _retrieval augmented generation (RAG)_, where the knowledge graph, itself, is built using a _large language model (LLM)_.

TLDRLC is an experimental software. It is not associated with any, and should not be interpreted as a, reliable chatbot service. The rationale behind this project is experimentation with  retrieval augmented generation. The implementation is based on concepts from publicly available tutorials and courses, such as [DeepLearning.ai: Knowledge Graphs for RAG](https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/) and [DeepLearning.ai: Building and Evaluating Advanced RAG](https://www.deeplearning.ai/courses/building-evaluating-advanced-rag/) to name a few.

## Installation

If you want to run this app in a Docker container then you can skip the installation of Python dependencies. The process of running the app with Docker is described below in the usage section. You may still need to setup the language model provider, and the storages for documents, indices and graphs.

### Python dependencies

You will need Python installed on your computer. The code in this repository has been tested on Python 3.12.0. Refrain from using the system Python. Use a Python version and virtual environment manager such as [pyenv](https://github.com/pyenv/pyenv). Create a new Python virtual environment for Python 3.12.0 or above. You can install all the dependencies for this application in your virtual environment by running the following.

```
pip install --upgrade pip
pip install -r requirements.txt
```
#### Optional: Uninstall all dependencies to start afresh

If necessary, you can uninstall everything previously installed by `pip` (in a virtual environment) by running the following.

```
pip freeze | xargs pip uninstall -y
```

#### Upgrading Python dependencies

The currently installed packages can be upgraded and the `requirements.txt` updated accordingly by running the following.

```
sed 's/==.*$//' requirements.txt | xargs pip install --upgrade
pip list -l --not-required --format=freeze > requirements.txt
```

Note that while `pip freeze` outputs all the installed packages (in your virtual environment), `pip list --not-required --format=freeze` outputs only those that you installed, excluding those that were automatically retrieved as dependencies of those packages.

### Language model providers, documents, index and graph storage

#### Supported language model providers

You can specify the language model provider to use by using the environment variable `LLM_PROVIDER`, which defaults to `Ollama`, if not specified. The supported language model providers are:

1. Cohere.
2. Ollama.
3. Open AI.

If using [Ollama](https://ollama.com/), you will also need to install it or, point the chatbot to a remotely hosted Ollama server. You also need to pull the Ollama model that you specify with `OLLAMA_MODEL` environment variable using `ollama pull <model-name>` (replace `<model-name>` with the actual model that you want to use) on your Ollama server. Check the [available Ollama models](https://ollama.com/library).

Open AI can be used by specifying an `OPENAI_API_KEY`, an `OPENAI_MODEL`, and by choosing `Open AI` as the `LLM_PROVIDER`. Follow [this link](https://platform.openai.com/account/api-keys) to get an Open AI API key. Similarly, Cohere can be used by specifying a `COHERE_API_KEY`, a `COHERE_MODEL` (which defaults to `command`), and by choosing `Cohere` as the `LLM_PROVIDER`. Follow [this link](https://cohere.com/pricing) to obtain a Cohere API key.

See the settings in the `.env.template` file customisation of the LLM settings.

#### Documents and index storage

A [Redis stack installation](https://redis.io/docs/install/install-stack/) is required to store the parsed document chunks and the index, which can be reloaded without having to rebuild the index from the sources. Although the default settings are okay, refer to the [Redis persistence configuration](https://redis.io/docs/management/persistence/), if necessary. Alternative to a local Redis installation, you can also opt for a remote Redis installation such as one on [Render](https://render.com/). Depending on your hosting plan, various limits will apply on such cloud-based managed Redis installations. 

#### Graph database

Similarly, you will need [Neo4j graph database](https://neo4j.com/). You can either install it locally or use a the [Neo4j Aura DB](https://neo4j.com/cloud/platform/aura-graph-database/), which is a cloud-hosted version of Neo4j.

### Performance evaluation using Langfuse

If you wish to use [Langfuse](https://langfuse.com/) for performance evaluation then set `EVAL_USE_LANGFUSE = "True"` in the `.env` file followed by `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_PRIVATE_KEY`, `LANGFUSE_URL`. You can set these up by running Langfuse self-hosted or by signing up and signing in to [the Langfuse cloud](https://cloud.langfuse.com/).

## Usage

### On your computer
Once you have installed the dependencies mentioned above in your Python virtual environment, to run the chatbot, execute `streamlit run Chatbot.py`. It will automatically open a browser (unless you have a headless terminal) to the chatbot application.

Some settings can be modified through the app's web interface. All settings can be modified by using a local `.env` file or as environment variables. A comprehensive list of the supported environment variables is available in the `.env.template` file in this repository, which can serve as a starting point.

### Docker
You can run the app in a Docker container. By default, the app inside the Docker container will not use persistable index storage, document storage or graph storage. To run the app in a Docker container, you have to build its image (which we name as `tldrlc` although you can choose any other name) and run an instance (which we name as `tldrlc-container` but you can also pick a name of your choice) of that image, as follows.

```
docker build  -t tldrlc .
docker create -p 8501:8501/tcp --name tldrlc-container tldrlc
docker container start tldrlc-container
```

Following this, the app will be accessible on your Docker host, for example as [http://localhost:8501](http://localhost:8501) -- assuming that nothing else on host was blocking port 8501.

If you want to change the settings of Streamlit or the app itself inside the container, login to the container as `root`. You can do this by running `docker exec -it tldrlc-container bash`. Once, you have the shell access in the container, edit the files `/app/.streamlit/config.toml` and `/app/.env` respectively using the `nano` editor that is installed for convenience. For example, you can change the default behaviour of the containerised app to use your preferred remote graph, index and document storage. Then, restart the _same_ container, by running `docker container restart tldrlc-container`. Remember that these changes _will not_ propagate to any new container that you spin out of the image.

### Cloud deployment

An example deployment of this app may be available on Render at [https://tldrlc.onrender.com](https://tldrlc.onrender.com). The cloud deployment on Render is for demonstrations only. The deployment may be spun down due to inactivity and it can take up to a minute to spin it up.

Another example deployment of this app may be available on the Streamlit community cloud at [https://tldrlc.streamlit.app](https://tldrlc.streamlit.app). Similar to the Render deployment, the Streamlit community cloud deployment may be spun down due to inactivity. If it is then you need to start it, in order to be able to interact with it.

For a cloud deployment, you have to use Open AI or Cohere. By default, graph, index and documents will be stored in memory with no disk persistence. If you want persistence with the deployment with a cloud deployment, you must use [Neo4j Aura DB](https://neo4j.com/cloud/platform/aura-graph-database/) and a remotely hosted Redis (e.g., [Redis on Render](https://docs.render.com/redis)). Alternatively, you can use local or on-premises hosted Ollama, Neo4j and Redis by exposing those services with TCP (or, TLS) tunnels publicly through [ngrok](https://ngrok.com/).

**Note** that for the aforementioned cloud deployment(s), unless you create your separate cloud deployment by yourself, LangFuse evaluation is enabled by default and cannot be turned off. This is meant for evaluation purposes of this project. Hence, all information about your interactions with the chatbot will be available to the maintainer(s) of this project. _For local, Docker or cloud deployments that you create, LangFuse is **not** enabled by default. Even when enabled, by yourself, LangFuse traces will be available to you, and not the maintainer(s) of this project_.

## Support

Use the issue tracker to report bugs or request feature enhancements. You can also use the discussions to discuss matters other than bug reports and feature requests.

## Contributing

You can contribute fixes or new ideas using pull requests.

## License

Apache License, Version 2.0, January 2004. See: [http://www.apache.org/licenses/](http://www.apache.org/licenses/).

## Project status

Actively maintained early stage prototype for experimentation only.
