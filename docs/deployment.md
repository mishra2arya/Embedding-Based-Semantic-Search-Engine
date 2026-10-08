# Deployment Guide: Docker & Kubernetes

## 1. Local Development
```bash
# Set up Python environment
make setup

# Run server with live reload
make serve
```

## 2. Docker Deployment
The production Dockerfile implements multi-stage compilation, non-root execution (`appuser:1001`), and healthcheck probes.

### Build & Run Container
```bash
# Build image
docker build -t semantic-search-engine:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e API_KEY_ADMIN=admin-secret-key-12345 \
  --name semantic-search \
  semantic-search-engine:latest
```

### Docker Compose
```bash
docker-compose up -d
```

## 3. Kubernetes Production Deployment
Pre-configured manifests are available in `k8s/`:
```bash
# Apply namespace
kubectl apply -f k8s/namespace.yaml

# Apply configuration and secrets
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.example.yaml

# Apply storage claim and deployment
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml
```

### Health & Readiness Probes
- **Liveness**: `GET /api/v1/health` (Initial delay: 20s, Period: 15s)
- **Readiness**: `GET /api/v1/ready` (Initial delay: 15s, Period: 10s)
