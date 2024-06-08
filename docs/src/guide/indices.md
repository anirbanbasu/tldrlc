# Documents, index and graph storage

## Documents and index storage

A [Redis stack installation](https://redis.io/docs/install/install-stack/) is required to store the parsed document chunks and the index, which can be reloaded without having to rebuild the index from the sources. Although the default settings are okay, refer to the [Redis persistence configuration](https://redis.io/docs/management/persistence/), if necessary. Alternative to a local Redis installation, you can also opt for a remote Redis installation such as one on [Render](https://render.com/). Depending on your hosting plan, various limits will apply on such cloud-based managed Redis installations. 

## Graph database

Similarly, you will need [Neo4j graph database](https://neo4j.com/). You can either install it locally or use a the [Neo4j Aura DB](https://neo4j.com/cloud/platform/aura-graph-database/), which is a cloud-hosted version of Neo4j.

_Note that unless you specify `NEO4J_DISABLE = "True"` to disable Neo4J and use an in-memory graph database, the Neo4J server must be accessible using the specified connection credentials. Otherwise, the application will display an error at the time of starting up._