FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /root/.streamlit && \
    echo "\
[server]\n\
address = \"0.0.0.0\"\n\
port = 8501\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
runOnSave = true\n\
\n\
[browser]\n\
gatherUsageStats = false\n" > /root/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "ai_03_streamlit.py"]
