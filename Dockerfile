FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app
ENV PDF_SOURCE_PATH=data/Hands_On_Machine_Learning_with_Scikit_Learn_and_TensorFlow.pdf
ENV FAISS_INDEX_DIR=storage/faiss_index

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8001

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8001", "--server.address=0.0.0.0"]
