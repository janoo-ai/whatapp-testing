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

ENV OpenAI_KEY "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"
ENV Pinecone_api_key "cc0325a0-6726-4ec4-8c31-0c9fd393ad6e"
ENV COHERE_API_KEY "FnFKnYEWxfCWiVV6R6o16MZtfdMqkS2dOlFYsA4p"
ENV WHATSAPP_ACCESS_TOKEN "EAAQoFmFf90IBOwQ8JLPeJqXQN8AcJPeERLURYAxeVE7lRsoCCEeeqSrwysw370JOk2uhSqjDP9HymSsFSVPAlefbcUP8mqWqz0PU1ySZCVIn4MLRXXKnKzwuXkfMKiVDSghv445QThPvB9deUDf3IanCwLDvZAnBtH5Tx5vu7OOP9qBZAZC1XGtZCePKSxg8lBgZDZD"


# Expose the port Flask runs on
EXPOSE 5004

# Command to run the application
CMD ["python", "app.py"]
