FROM ubuntu:24.10

RUN apt-get update
RUN apt-get install -y \
    python3 \
    python3-pip \
    python3-pandas \
    python3-matplotlib \
    python3-seaborn \
    python3-sklearn \
    python3-xgboost

RUN pip install --break-system-packages \
    shap

WORKDIR /root/src
COPY . .

CMD [ "python3", "main.py" ]
