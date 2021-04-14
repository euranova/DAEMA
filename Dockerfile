from python:3-slim
COPY missing_data_imputation/requirements.txt .
RUN pip install -r requirements.txt
COPY missing_data_imputation /missing_data_imputation
ENV PYTHONPATH "${PYTONPATH}:/missing_data_imputation/src"
WORKDIR /missing_data_imputation/src

