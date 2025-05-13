<<<<<<< HEAD
FROM python:3.9-slim

# Prevent prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install essential dependencies and SSL certs
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    libssl-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip securely
RUN python -m pip install --upgrade pip setuptools wheel

# Set the working directory
WORKDIR /app

# Copy only requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose port for Flask
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
=======
FROM python:3.9-slim

# Prevent prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install essential dependencies and SSL certs
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    libssl-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip securely
RUN python -m pip install --upgrade pip setuptools wheel

# Set the working directory
WORKDIR /app

# Copy only requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose port for Flask
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
>>>>>>> 979b5792c8e411302ee811c63003780bb660d43e
