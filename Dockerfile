FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so editing the examples does not reinstall everything.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[anthropic,redis]"

COPY examples ./examples

RUN useradd --create-home --uid 10001 router
USER router

EXPOSE 8080
# Mount your own config here, or set LLM_ROUTER_CONFIG to another path.
ENV LLM_ROUTER_CONFIG=/config/router.yaml
ENTRYPOINT ["llm-router", "--host", "0.0.0.0", "--port", "8080"]
