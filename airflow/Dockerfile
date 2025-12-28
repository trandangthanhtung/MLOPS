FROM apache/airflow:2.9.3-python3.9

USER root
RUN apt-get update && apt-get install -y gcc build-essential \
    && apt-get clean

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
