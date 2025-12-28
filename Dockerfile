FROM python:3.9-slim

WORKDIR /app

# Copy and install requirements
COPY requirements_hf.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Expose Hugging Face default port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]