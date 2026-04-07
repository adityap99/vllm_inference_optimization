#!/bin/bash
set -e

# Remove existing containers if they exist
podman rm -f prometheus grafana 2>/dev/null || true

# Run Prometheus
echo "Starting Prometheus..."
podman run -d --name prometheus \
    --network host \
    --userns=keep-id \
    --user $(id -u):$(id -g) \
    -v ./prometheus_pd_host.yaml:/etc/prometheus/prometheus.yml \
    prom/prometheus:latest

# Run Grafana
echo "Starting Grafana..."
podman run -d --name grafana \
    --network host \
    --userns=keep-id \
    --user $(id -u):$(id -g) \
    grafana/grafana:latest

echo "Prometheus and Grafana started with host networking, keep-id, and current user."
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000"
echo "Note: When configuring the Prometheus data source in Grafana, use URL: http://localhost:9090"
