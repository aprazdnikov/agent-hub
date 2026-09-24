# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.12.15 AS uv

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

FROM python:3.12-slim-bookworm
# Match the host user so files the agent writes into the mounted workspace keep your ownership.
ARG UID=1000
ARG GID=1000
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${GID}" app \
    && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash app \
    && mkdir -p /workspace /data /home/app/.claude \
    && chown app:app /workspace /data /home/app/.claude
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
