# Multi-stage build for lrcfilter
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies (ffmpeg not needed here - only for runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy application code first for dependency installation
COPY --chown=root:root . /app
RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime
FROM python:3.11-slim as runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN useradd -m -u 1000 lrcfilter
USER lrcfilter
WORKDIR /home/lrcfilter

# Copy application code (after dependencies are installed for better caching)
COPY --chown=lrcfilter:lrcfilter . .

# Install the package
RUN pip install --no-cache-dir -e .

# Default entrypoint
ENTRYPOINT ["python", "-m", "lrcfilter"]
CMD ["--help"]
