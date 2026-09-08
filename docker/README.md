# Docker stacks

Compose files and Dockerfiles for the containerized OpenMMLA services:

- `docker-compose.asr.yml`, `docker-compose.vfa.yml`: the ASR and VFA AI service stacks, one image per service (Dockerfiles in the sibling directories).
- `docker-compose.infra.yml` with `.env.example`: the database stack, InfluxDB + MongoDB for the uber server.

Setup, migration, backups, MongoDB authentication, host-port overrides and the known caveats are documented in [docs/docker.md](../docs/docker.md).
