# Start from an official Python 3.11 image
# "slim" variant = smaller size, no unnecessary OS packages
FROM python:3.11-slim

# Set the working directory inside the container
# All subsequent commands run from here
WORKDIR /app

# Copy requirements first — before copying the rest of the code
# Docker builds in layers. If requirements.txt didn't change,
# Docker reuses the cached layer and skips reinstalling packages.
# This makes rebuilds much faster.
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir tells pip not to store the download cache
# This keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project into the container
COPY . .

# Create the data directory in case it doesn't exist
# -p means create parent directories too, and don't error if it exists
RUN mkdir -p data/letters

# Tell Docker this container listens on port 8501
# This is Streamlit's default port
EXPOSE 8501

# The command that runs when the container starts
# --server.address=0.0.0.0 makes Streamlit accessible from outside the container
# Without this it only listens on localhost inside the container
# --server.port=8501 is explicit about the port
# --server.headless=true disables the "open browser" prompt since there's no browser in a container
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]