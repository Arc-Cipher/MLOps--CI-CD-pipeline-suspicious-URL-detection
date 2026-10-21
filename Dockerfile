FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY features.py train.py app.py phishing_urls_1000.csv ./
COPY templates templates

# Create directory, user, and set ownership
RUN mkdir -p /srv/data && \
    useradd -m app && \
    chown -R app:app /srv /srv/data

# Switch to non-root user before running training
USER app

# Bake a freshly trained model into the image as the app user
RUN python train.py

EXPOSE 5000

HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:5000/health')"

CMD ["gunicorn", "-w", "1", "--threads", "8", "-b", "0.0.0.0:5000", "app:app"]