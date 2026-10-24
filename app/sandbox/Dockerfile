
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /sandbox

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install sandbox dependencies
RUN uv venv /sandbox/.venv \
    && uv pip install --python /sandbox/.venv/bin/python \
       pandas numpy pyarrow

ENV PATH="/sandbox/.venv/bin:$PATH"

COPY app/sandbox/runner.py /sandbox/runner.py

RUN useradd --create-home --uid 10001 sandboxuser

USER sandboxuser

ENTRYPOINT ["python", "/sandbox/runner.py"]
