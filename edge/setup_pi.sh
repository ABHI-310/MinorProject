#!/bin/bash
# ==============================================================================
# setup_pi.sh
# Automated Provisioning Script for Raspberry Pi 4 (64-bit OS)
# Prepares edge environment for ONNX runtime inference & Pi Camera Module 2
# ==============================================================================

set -e

echo "=== Initializing Raspberry Pi 4 for PCB Smart Maintenance Edge Inference ==="

# 1. Update system packages
echo "Updating apt repositories..."
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install essential system dependencies and build libraries
echo "Installing system vision and communication libraries..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    python3-numpy \
    libcamera-dev \
    libcamera-apps \
    libatlas-base-dev \
    libcap-dev \
    git \
    curl

# 3. Upgrade pip and install edge Python packages
echo "Installing Python edge runtime packages..."
pip3 install --upgrade pip
pip3 install -r requirements_pi.txt

# 4. Configure project directories
EDGE_DIR="/home/pi/pcb_maintenance"
echo "Creating application directory: $EDGE_DIR"
mkdir -p "$EDGE_DIR"
mkdir -p "$EDGE_DIR/weights"

# 5. Provide instructions for Camera Module 2
echo "----------------------------------------------------------------------"
echo "Camera Configuration:"
echo "Ensure Camera Module 2 ribbon cable is firmly seated in the CSI port."
echo "Enable Camera interface if needed via: sudo raspi-config -> Interface Options -> Camera"
echo "----------------------------------------------------------------------"

# 6. Create systemd service for automated daemon execution
SERVICE_FILE="/etc/systemd/system/pcb_detection.service"
echo "Registering systemd service at $SERVICE_FILE..."

sudo bash -c "cat > $SERVICE_FILE" << EOF
[Unit]
Description=PCB Defect Detection Edge Inference Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/pcb_maintenance
ExecStart=/usr/bin/python3 /home/pi/pcb_maintenance/edge_inference.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo "Enabling service to launch on system boot..."
sudo systemctl enable pcb_detection.service

echo "✓ Raspberry Pi 4 Edge Setup Complete!"
echo "To start edge inference manually: python3 edge_inference.py"
echo "To start as background daemon: sudo systemctl start pcb_detection"

