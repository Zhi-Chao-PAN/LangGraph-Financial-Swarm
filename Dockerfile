# Build Stage
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Final Stage
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source code
# Copy source code
COPY src/ ./src/
COPY main.py ./

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1

# Entrypoint
CMD ["python", "main.py"]
