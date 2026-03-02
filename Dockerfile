###############################################################################
# Dockerfile — Reproducible Spark environment for Fake News Detection
# Module: 7006SCN Machine Learning and Big Data (Coventry University)
#
# Build:  docker build -t fakenews-bigdata .
# Run:    docker run -p 8888:8888 -p 4040:4040 -v ${PWD}:/app fakenews-bigdata
#           Port 8888 = JupyterLab   |   Port 4040 = Spark UI
###############################################################################
FROM python:3.10-slim-bookworm

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        curl wget git procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ---------- Spark standalone (matches pyspark wheel) ----------
ENV SPARK_VERSION=3.5.1
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
RUN curl -fsSL "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    | tar -xz -C /opt && mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME}
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"

# ---------- AWS SDK JARs for S3 access ----------
RUN curl -fsSL -o ${SPARK_HOME}/jars/hadoop-aws-3.3.4.jar \
        https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    curl -fsSL -o ${SPARK_HOME}/jars/aws-java-sdk-bundle-1.12.367.jar \
        https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.367/aws-java-sdk-bundle-1.12.367.jar

# ---------- Python packages ----------
WORKDIR /app
COPY environment.yml .
RUN pip install --no-cache-dir \
        pyspark==3.5.1 pyarrow==15.0.0 findspark==2.0.1 \
        scikit-learn==1.4.0 pandas==2.2.0 numpy==1.26.4 scipy==1.12.0 \
        matplotlib==3.8.3 seaborn==0.13.2 \
        boto3==1.34.40 warcio==1.7.4 beautifulsoup4==4.12.3 lxml==5.1.0 \
        jupyterlab==4.1.1 ipykernel==6.29.2 nbformat==5.9.2 \
        tqdm==4.66.1 pyyaml==6.0.1 python-dotenv==1.0.1 \
        pytest==8.0.0 pytest-cov==4.1.0

COPY . .

EXPOSE 8888 4040

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
