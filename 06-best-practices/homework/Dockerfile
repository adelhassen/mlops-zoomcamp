FROM python:3.9.12-slim

RUN pip install -U pip & pip install pipenv

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "batch.py", "batch.py" ]
COPY [ "model.bin", "model.bin" ]

ENTRYPOINT [ "python", "batch.py" ]