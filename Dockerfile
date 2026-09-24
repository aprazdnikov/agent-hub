# syntax=docker/dockerfile:1

# Toolchain versions for the `dev` target; override with --build-arg.
ARG JAVA_VERSION=25
ARG MAVEN_VERSION=3.9
ARG NODE_VERSION=24
ARG DOCKER_VERSION=29

FROM ghcr.io/astral-sh/uv:0.12.15 AS uv
FROM eclipse-temurin:${JAVA_VERSION}-jdk AS jdk
FROM maven:${MAVEN_VERSION} AS maven
FROM node:${NODE_VERSION}-bookworm-slim AS node
FROM docker:${DOCKER_VERSION}-cli AS docker

FROM python:3.12-slim-bookworm AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# Dependencies first: this layer is rebuilt only when the lock file changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# Bot + Claude Code + git: enough for Python and plain-text projects.
FROM python:3.12-slim-bookworm AS base
# Match the host user so files the agent writes into the mounted workspace keep your ownership.
ARG UID=1000
ARG GID=1000
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${GID}" app \
    && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash app \
    && mkdir -p /workspace /data /home/app/.claude /home/app/.m2 /home/app/.npm \
    && chown app:app /workspace /data /home/app/.claude /home/app/.m2 /home/app/.npm
COPY --from=build --chown=app:app /app/.venv /app/.venv
# The SDK ships its own Claude Code binary; expose it for `claude /login` inside the container.
RUN ln -s /app/.venv/lib/python3.12/site-packages/claude_agent_sdk/_bundled/claude /usr/local/bin/claude
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    CLAUDE_CONFIG_DIR=/home/app/.claude \
    AGENT_HUB_WORKSPACE_ROOT=/workspace \
    AGENT_HUB_STATE_FILE=/data/topics.json
USER app
WORKDIR /workspace
ENTRYPOINT ["agent-hub"]

# base + JDK, Maven, Node.js, npm and the Docker CLI for JVM and JavaScript projects.
# The Docker CLI talks to the host daemon only when compose.docker.yaml mounts its socket.
FROM base AS dev
USER root
COPY --from=jdk /opt/java/openjdk /opt/java/openjdk
COPY --from=maven /usr/share/maven /usr/share/maven
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=docker /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker /usr/local/libexec/docker/cli-plugins /usr/local/libexec/docker/cli-plugins
RUN ln -s /usr/share/maven/bin/mvn /usr/local/bin/mvn \
    && ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && ln -s ../lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack
ENV JAVA_HOME=/opt/java/openjdk \
    MAVEN_HOME=/usr/share/maven \
    PATH=/opt/java/openjdk/bin:$PATH
USER app
