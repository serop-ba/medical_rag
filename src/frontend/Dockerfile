# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY ./requirements.txt /app/requirements.txt

# Install any dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app

# Command to run the Streamlit app
CMD ["streamlit", "run", "./src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
