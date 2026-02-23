
#!/usr/bin/env bash
set -euo pipefail

RUNS_DIR="/workspace/CTMP_GIN/runs"
REMOTE_BASE="gdrive:CTMP_GIN_runs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REMOTE="$REMOTE_BASE/$TIMESTAMP"

echo "===== upload start ====="
echo "source: $RUNS_DIR"
echo "dest  : $REMOTE"

# runs 폴더 존재 확인
if [ ! -d "$RUNS_DIR" ]; then
  echo "runs directory not found"
  exit 1
fi

# 업로드 (copy: 원격 기존 파일 삭제 없이 추가/덮어쓰기만)
rclone copy "$RUNS_DIR" "$REMOTE" \
  --create-empty-src-dirs \
  --transfers 8 \
  --checkers 16 \
  --retries 5 \
  --low-level-retries 10 \
  --stats 10s

echo "===== upload done ====="

# 업로드 성공 확인 (목적지 접근 가능 여부)
if ! rclone lsf "$REMOTE" >/dev/null 2>&1; then
  echo "upload verification failed"
  exit 2
fi

echo "===== upload verified ====="

touch "$RUNS_DIR/UPLOAD.ok"