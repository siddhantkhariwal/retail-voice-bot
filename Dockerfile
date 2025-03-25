# Use the official Python image as a base
FROM python:3.10-slim

# Set the working directory for the application
WORKDIR /app

# Copy the requirements file to leverage caching
COPY requirements.txt .

# Install the necessary Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Streamlit runs on for accessibility
EXPOSE 8501

# Create a non-root user 
RUN useradd -m appuser

# Change ownership of the application files to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Use ENTRYPOINT to run Streamlit
ENTRYPOINT ["streamlit", "run", "main.py"]

# Default arguments for ENTRYPOINT
CMD ["--server.port=8501", "--server.address=0.0.0.0"]                                                            
