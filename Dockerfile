from python:3-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /daema
ENV PYTHONPATH "${PYTONPATH}:/daema/src"
WORKDIR /daema/src
