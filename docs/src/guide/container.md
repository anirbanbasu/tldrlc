# Container (Docker)
You can run the app in a Docker container. By default, the app inside the Docker container will not use persistable index storage, document storage or graph storage. To run the app in a Docker container, you have to build its image (which we name as `tldrlc` although you can choose any other name) and run an instance (which we name as `tldrlc-container` but you can also pick a name of your choice) of that image, as follows.

```
docker build -f local.dockerfile -t tldrlc .
docker create -p 8765:8765/tcp --name tldrlc-container tldrlc
```

<!-- If you want to change the settings of the app itself inside the container, login to the container as `root`. You can do this by running `docker exec -it tldrlc-container bash`. Once, you have the shell access in the container, edit the file `/app/.env` using the `nano` editor that is installed for convenience. For example, you can change the default behaviour of the containerised app to use your preferred remote graph, index and document storage. Then, restart the _same_ container, by running `docker container restart tldrlc-container`. Remember that these changes _will not_ propagate to any new container that you spin out of the image. -->