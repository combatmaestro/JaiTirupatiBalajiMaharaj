FROM python:3.10-slim

# Set working dir
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models during build (optional - or you can do this in code)
RUN mkdir -p models/buffalo_l
RUN curl -L -o inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx

# Optional: download buffalo_l files
# RUN curl -L -o models/buffalo_l/det_10g.onnx https://some-url...

# Copy all source files
COPY . .

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
