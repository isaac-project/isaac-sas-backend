FROM tiangolo/meinheld-gunicorn:python3.7
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
