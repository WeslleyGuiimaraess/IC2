FROM python:3.8-bookworm

RUN apt-get update && apt-get install -y python3-opengl xauth
COPY requirements.txt .

RUN pip install -r requirements.txt