# Run locally

## The non-containerised way

Once you have installed the dependencies mentioned above in your Python virtual environment, to run the chatbot, execute `solara run app.py`. It will automatically open a browser (unless you have a headless terminal) to the chatbot application. An alternative way of running the app is by executing the script `run_starlette.sh`, which will load the app using the [Starlette framework](https://www.starlette.io/) on the Asynchronous Server Gateway Interface (ASGI) server, [uvicorn](https://www.uvicorn.org/).

## Containerised (Docker)

Following the [creation of the container](container.md), you can run the app using `docker container start tldrlc-container` the app will be accessible on your Docker host, for example as [http://localhost:8765](http://localhost:8765) -- assuming that nothing else on host is blocking port 8765 when the container starts.

The Docker container has to depend on external LLM provider, graph database, document and index storage. If any of these, such as `Ollama`, is running on the Docker host then you should use the host name for the service as `host.docker.internal` or `gateway.docker.internal`. See [the networking documentation of Docker desktop](https://docs.docker.com/desktop/networking/) for details.