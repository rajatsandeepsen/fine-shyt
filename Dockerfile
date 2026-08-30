FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy entire project into the image
COPY . /server
WORKDIR /server

ENV UV_NO_DEV=1
RUN uv sync --locked

CMD ["uv", "run", "fastapi", "run", "--port", "3000"]
