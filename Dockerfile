FROM python:3.11.1

WORKDIR ..
COPY . /app/
WORKDIR /app

# Install app requirements.
RUN pip install -r requirements.txt

ENTRYPOINT [ "python3", "manage.py" ]