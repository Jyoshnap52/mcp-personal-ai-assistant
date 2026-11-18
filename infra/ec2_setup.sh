#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "  MCP Assistant Production Provisioning   "
echo "=========================================="

# 1. Update OS and install essential tools
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release git

# 2. Install Docker Engine and Docker Compose Plugin
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Enable and start Docker service
    sudo systemctl enable docker
    sudo systemctl start docker

    # Add current user to docker group
    sudo usermod -aG docker "$USER"
    echo "Docker installed successfully."
fi

# 3. Verify Docker Compose
docker compose version

echo "=========================================="
echo "  Provisioning Complete!                  "
echo "  To launch:                             "
echo "    1. Create your .env file with secrets "
echo "    2. Run: docker compose -f docker-compose.prod.yml up -d --build"
echo "=========================================="
