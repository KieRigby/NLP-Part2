FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY ../api /app

RUN mkdir /app/utils

COPY ../utils /app/utils

RUN pip install -r requirements.txt

EXPOSE 9090

RUN [ "python", "-c", "import nltk; nltk.download('stopwords'); nltk.download('wordnet')" ]

