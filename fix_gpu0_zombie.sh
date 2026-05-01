#!/usr/bin/env bash
# Run this on the HOST (not inside container) to free GPU 0 zombie memory

echo "=== GPU 0 상태 ==="
nvidia-smi -i 0 --query-gpu=memory.used,memory.free --format=csv

echo ""
echo "=== Exited 컨테이너 목록 ==="
docker ps -a --filter status=exited

echo ""
echo "=== Exited 컨테이너 제거 (GPU 메모리 해제) ==="
docker container prune -f

echo ""
echo "=== 제거 후 GPU 0 상태 ==="
nvidia-smi -i 0 --query-gpu=memory.used,memory.free --format=csv
