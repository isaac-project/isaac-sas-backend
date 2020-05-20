FROM tiangolo/meinheld-gunicorn-flask:python3.7
ADD . /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

