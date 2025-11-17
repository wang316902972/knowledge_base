#!/bin/bash

# 健康检查脚本

set -e

# 获取端口，默认8001
PORT=${API_PORT:-8001}

# 执行健康检查
python3 -c "
import requests
import sys
import os

try:
    port = os.environ.get('API_PORT', '8001')
    url = f'http://localhost:{port}/health'
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    print('Health check passed')
    sys.exit(0)
except Exception as e:
    print(f'Health check failed: {e}')
    sys.exit(1)
"