FROM python:3.11-slim
WORKDIR /opt/spire
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY spire/ spire/
COPY train.py predict.py ./
ENV TF_USE_LEGACY_KERAS=1
CMD ["python", "train.py", "--help"]
