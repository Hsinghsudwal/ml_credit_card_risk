
FROM python:3.10-slim


WORKDIR /monitoring


COPY requirements.txt /monitoring/


RUN pip install --upgrade pip


RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir localstack-client boto3 evidently



COPY . /monitoring/



EXPOSE 8501


ENV PYTHONUNBUFFERED=1


CMD ["streamlit", "run", "app.py"]

