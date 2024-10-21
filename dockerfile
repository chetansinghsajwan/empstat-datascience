FROM ubuntu:24.10

RUN apt-get update
RUN apt-get install -y \
    python3 \
    python3-pandas \
    python3-matplotlib \
    python3-seaborn \
    python3-sklearn

WORKDIR /root/src
COPY . .

CMD [ "python3", "main.py" ]
