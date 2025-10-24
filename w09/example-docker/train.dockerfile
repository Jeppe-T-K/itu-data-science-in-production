FROM python:3.9

WORKDIR /usr/local/app

# Install the application dependencies
COPY . ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY example_docker ./src

# Setup an app user so the container doesn't run as the root user

CMD ["python", "src/modeling/train.py"]