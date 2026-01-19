FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Copy only dependency definition files
COPY pyproject.toml uv.lock* ./ 

# Install uv
RUN pip install uv

# Install project dependencies (excluding dev dependencies if used)
RUN uv sync --no-dev

# Copy application source code and data
COPY src ./src
COPY app ./app
COPY data ./data

# Ensure Python can locate packages under src
ENV PYTHONPATH=/app/src

# Expose the Streamlit port
EXPOSE 8501

# Default command: start the Streamlit application
CMD ["uv", "run", "streamlit", "run", "app/kurkkumopo_app.py", "--server.port=8501", "--server.address=0.0.0.0"]