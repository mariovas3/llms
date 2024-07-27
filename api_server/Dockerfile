FROM python:3.11-slim

WORKDIR /app

# copy files of saved_models api_server and src;
COPY . ./app

# will install the torch-cpu dependencies manually so remove
# their entries from the requirements file;
RUN sed -i -E 's/.*torch.*|.*lightning.*//' ./app/requirements.txt && \
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install lightning==2.3.3 tiktoken==0.7.0 transformers==4.43.2 flask==3.0.3

ENV PYTHONPATH ./app

# Make port 5000 available to the world outside this container
EXPOSE 5000

CMD ["python", "./app/api_server/app.py"]