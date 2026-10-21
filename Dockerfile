FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY features.py train.py app.py ./
COPY templates templates
# Bake a freshly trained, quality-gated model into the image
RUN python train.py
RUN useradd -m app && chown -R app /srv
USER app
EXPOSE 5000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:5000/health')"
# One worker keeps the in-memory drift window consistent; threads give concurrency
CMD ["gunicorn", "-w", "1", "--threads", "8", "-b", "0.0.0.0:5000", "app:app"]
