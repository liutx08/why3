#!/bin/bash
set -e  # 如果有错误就终止脚本执行

echo "Updating system and installing dependencies..."
apt-get update && apt-get install -y openbabel swig

echo "Installing Python dependencies..."
pip install -r requirements.txt
