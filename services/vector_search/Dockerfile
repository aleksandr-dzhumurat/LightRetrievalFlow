FROM ubuntu:22.04

ENV PYTHONPATH=/srv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv/

COPY services/vector_search/requirements.txt /srv/

RUN \
    apt update && \
    apt install software-properties-common -y && \
    add-apt-repository 'ppa:deadsnakes/ppa' && \
    apt-get update && \
    apt install -y python3.10 && \
    apt install -y python3-pip

RUN \
    python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install --no-cache -r requirements.txt

COPY services/vector_search/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]