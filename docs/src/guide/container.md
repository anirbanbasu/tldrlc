# Container (Docker)
You can run the app in a Docker container. By default, the app inside the Docker container will not use persistable index storage, document storage or graph storage. To run the app in a Docker container, you have to build its image (which we name as `tldrlc` although you can choose any other name) and run an instance (which we name as `tldrlc-container` but you can also pick a name of your choice) of that image, as follows.

```
docker build -f local.dockerfile -t tldrlc .
docker create -p 8765:8765/tcp --name tldrlc-container tldrlc
docker container start tldrlc-container
```