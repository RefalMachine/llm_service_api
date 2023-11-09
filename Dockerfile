FROM python:3.10

COPY requirements.txt /workdir/

WORKDIR /workdir
RUN pip install -r requirements.txt

COPY src/ /workdir/src/
COPY app/ /workdir/app/
COPY run_server.py /workdir/

CMD [ "python", "./run_server.py"]