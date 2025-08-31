FROM python:3.12-slim

WORKDIR /app

# Copy requirements first (for better caching)
COPY requirement.txt .


RUN pip install -r requirement.txt

# Copy the rest of the app
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app_frontend.py"]
