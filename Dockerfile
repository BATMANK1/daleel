# The web service: the code and the libraries of the two models. The data it
# answers from, the models' weights and the API key all come from outside the
# image, at run time (compose.yaml): the data is derived from documents that
# are not redistributable, and the key is a secret.
#
#   docker build -t daleel:cpu .
#   docker build -t daleel:gpu --build-arg TORCH_BUILD=cu130 .

FROM python:3.12-slim

# PyTorch's build: cpu, or a CUDA version the host's driver supports (cu130 needs
# a driver for CUDA 13). The version is the one the evaluation ran with.
ARG TORCH_BUILD=cpu
ARG TORCH_VERSION=2.14.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/daleel/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

# The service runs as an ordinary user. Its id is the usual first user's on
# Linux, so that it can write to ./data/serve when that is mounted from the host;
# where yours differs (id -u), build with --build-arg UID=<yours>.
ARG UID=1000
RUN useradd --create-home --uid "${UID}" daleel \
    && mkdir -p /home/daleel/.cache/huggingface \
    && chown -R daleel:daleel /home/daleel/.cache
WORKDIR /app

# PyTorch first, from the index of the chosen build, so that the models'
# libraries find it installed and do not pull another build from PyPI.
RUN pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_BUILD}"

# The dependencies alone, against an empty package, so that this layer is
# rebuilt only when pyproject.toml changes, not with every change to the code.
COPY pyproject.toml ./
RUN mkdir -p src/daleel && touch src/daleel/__init__.py \
    && pip install ".[dense]" \
    && pip uninstall -y daleel \
    && rm -rf src

COPY src ./src
RUN pip install --no-deps . && rm -rf src

USER daleel
EXPOSE 8000
# Loading the models takes a minute, and downloading them on the first start longer.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["daleel", "serve", "--host", "0.0.0.0", "--port", "8000"]
