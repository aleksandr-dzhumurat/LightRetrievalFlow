# Getting started

- [ ]  rename `.env.template` -> `.env`. Fill `CLICKHOUSE_LOGIN`, `CLICKHOUSE_PASS`, `ELASTIC_USERNAME`, `ELASTIC_PASSWORD`
- [ ]  run `make prepare-dirs`
- [ ]  prepare python env
- [ ]  enjoy!

## Python env

```shell
pyenv install 3.11 && \
pyenv virtualenv 3.11 ltr-research
```

```shell
source ~/.pyenv/versions/ltr-research/bin/activate && \
pip install --upgrade pip && \
pip install -r requirements.txt
```

Inslall current repo
```shell
pip install -e .
```

```shell
make inference
```