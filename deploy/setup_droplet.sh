#!/usr/bin/env bash
# One-time provisioning for a fresh Ubuntu 24.04 DigitalOcean Droplet (run as root).
#   ssh root@<droplet-ip> 'bash -s' < deploy/setup_droplet.sh
# Installs Docker + Compose. After this, push the repo up and `docker compose up -d --build`.
set -euo pipefail

echo "==> apt update + Docker"
apt-get update
apt-get install -y ca-certificates curl gnupg rsync
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker

# 4 GB RAM is tight while SigLIP+CLAP+Whisper+BGE warm up together — add swap as a safety net.
if [ ! -f /swapfile ]; then
  echo "==> adding 4G swap"
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "==> done. Next, from your laptop:"
echo "    rsync -az --filter=':- .dockerignore' ./ root@<ip>:/opt/shouldipost/"
echo "    ssh root@<ip> 'cd /opt/shouldipost/deploy && SITE_ADDRESS=your.domain docker compose up -d --build'"
