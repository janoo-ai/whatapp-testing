# Use the Python 3.9 slim image instead of alpine
FROM python:3.12.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for Python packages (e.g., psycopg2, scikit-learn)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    build-essential \
    pkg-config \
    libmariadb-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and the rest of the application
COPY . /app

# Upgrade pip and install the Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirement.txt

# Expose the port Flask runs on
EXPOSE 5004

# Command to run the application
CMD ["python", "app.py"]
