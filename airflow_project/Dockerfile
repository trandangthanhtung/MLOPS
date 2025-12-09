FROM apache/airflow:2.9.3

COPY requirements.txt /tmp/requirements.txt

USER airflow
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt