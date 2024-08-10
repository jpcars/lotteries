FROM python:3.10-slim

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  LC_ALL=C.UTF-8 \
  LANG=C.UTF-8 \
  LANGUAGE=C.UTF-8 \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.3.1

# System deps:
RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /usr/local/
COPY pyproject.toml README.md .
COPY lotteries ./lotteries

RUN apt-get update && apt-get install -y \
    make \
    curl \
    gcc

# Project initialization:
RUN poetry config virtualenvs.create false \
  && poetry install --only main --no-interaction --no-ansi


EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit/Rescue_Lotteries.py", "--server.port=8501", "--server.address=0.0.0.0"]
