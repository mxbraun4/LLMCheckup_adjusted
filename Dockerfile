# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04 AS runtime

# 1️⃣ rarely-changing system + Python deps  (cached)
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv python3-dev gcc build-essential git pkg-config default-libmysqlclient-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# 2️⃣ Python wheels
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r requirements.txt mysqlclient


# 3️⃣ Extras that seldom change
RUN git clone --depth 1 https://github.com/bigscience-workshop/petals.git /tmp/petals && \
    python3 -m pip install /tmp/petals && \
    rm -rf /tmp/petals && \
    python3 -m nltk.downloader omw-1.4 punkt && \
    python3 -m pip install --no-cache-dir -U spacy && \
    python3 -m spacy download en_core_web_sm

# 4️⃣ Your rapidly changing source code
COPY . .

ENV PORT=4000
EXPOSE ${PORT}
CMD ["gunicorn", "--timeout", "0", "-b", ":4000", "flask_app:app"]
