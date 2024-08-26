# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy and install requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Add the current directory contents into the container at /app
ADD . /app/

# Default env variables
# os.environ["OPENAI_BASE_URL"] = 'http://172.29.4.22:8000/v1'
# os.environ["OPENAI_API_KEY"] = "L3n0v0AI"
# os.environ['WHISPER_API_URL'] = "http://localhost:8001"

# Make port 80 available to the world outside this container
EXPOSE 8080

# Run python script
CMD ["python", "main.py"]
