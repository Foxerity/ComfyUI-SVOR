#!/usr/bin/env bash
set -euo pipefail

# Download the SVOR LoRA weights with wget only, avoiding huggingface-cli.
# Usage:
#   bash download_weights_wget.sh [models_dir]
#
# Optional environment variables:
#   HF_TOKEN      Hugging Face token, useful if your IP/account is rate-limited.
#   HF_ENDPOINT   Hugging Face endpoint. Default: https://huggingface.co
#   HF_REVISION   Model repo revision. Default: main
#   WGET_BIN      wget executable. Default: wget
#   LIMIT_RATE    Optional wget rate limit, for example 20m.
#   DRY_RUN       Set to 1 to print planned downloads without fetching weights.
#   QUIET         Set to 1 to hide wget progress output.

MODEL_DIR="${1:-models}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_ENDPOINT="${HF_ENDPOINT%/}"
HF_REVISION="${HF_REVISION:-main}"
WGET_BIN="${WGET_BIN:-wget}"
LIMIT_RATE="${LIMIT_RATE:-}"
DRY_RUN="${DRY_RUN:-0}"
QUIET="${QUIET:-0}"

SVOR_REPO="HigherHu/SVOR"

USER_AGENT="${USER_AGENT:-Mozilla/5.0 (compatible; SVOR-wget-downloader/1.0; +https://huggingface.co/HigherHu/SVOR)}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  elif command -v python >/dev/null 2>&1; then
    echo python
  else
    echo "Error: python3 or python is required to parse Hugging Face repo metadata." >&2
    exit 1
  fi
}

require_command "$WGET_BIN"
PYTHON_BIN="$(find_python)"

WGET_COMMON_ARGS=(
  "--continue"
  "--tries=20"
  "--waitretry=20"
  "--read-timeout=120"
  "--timeout=60"
  "--user-agent=${USER_AGENT}"
)

if [[ "$QUIET" == "1" ]]; then
  WGET_COMMON_ARGS+=("--no-verbose")
else
  WGET_COMMON_ARGS+=("--show-progress" "--progress=bar:force:noscroll")
fi

if "$WGET_BIN" --help 2>/dev/null | grep -q -- "--retry-on-http-error"; then
  WGET_COMMON_ARGS+=("--retry-on-http-error=429,500,502,503,504")
fi

if [[ -n "${LIMIT_RATE}" ]]; then
  WGET_COMMON_ARGS+=("--limit-rate=${LIMIT_RATE}")
fi

mkdir -p "$MODEL_DIR"

download_file() {
  local url="$1"
  local output_path="$2"
  local expected_size="${3:-}"
  local auth_scope="${4:-public}"
  local wget_args=("${WGET_COMMON_ARGS[@]}")
  local backup_path=""

  if [[ "$auth_scope" == "hf" && -n "${HF_TOKEN:-}" ]]; then
    wget_args+=("--header=Authorization: Bearer ${HF_TOKEN}")
  fi

  mkdir -p "$(dirname "$output_path")"

  if [[ -n "$expected_size" && "$expected_size" != "None" && "$expected_size" != "null" && -f "$output_path" ]]; then
    local existing_size
    existing_size="$(wc -c < "$output_path" | tr -d '[:space:]')"
    if [[ "$existing_size" == "$expected_size" ]]; then
      echo "Already complete: ${output_path} (${existing_size} bytes)"
      return 0
    fi
    if [[ "$existing_size" -gt "$expected_size" ]]; then
      backup_path="${output_path}.bad.$(date +%Y%m%d%H%M%S)"
      echo "Existing file is larger than expected; moving it aside:"
      echo "  ${output_path} -> ${backup_path}"
      mv "$output_path" "$backup_path"
    fi
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    if [[ -n "$expected_size" ]]; then
      echo "Would download: ${output_path} (${expected_size} bytes)"
    else
      echo "Would download: ${output_path}"
    fi
    echo "  from: ${url}"
    return 0
  fi

  echo "Downloading: ${output_path}"
  "$WGET_BIN" "${wget_args[@]}" \
    --output-document="$output_path" \
    "$url"

  if [[ ! -s "$output_path" ]]; then
    echo "Error: downloaded file is empty: ${output_path}" >&2
    return 1
  fi

  if [[ -n "$expected_size" && "$expected_size" != "None" && "$expected_size" != "null" ]]; then
    local actual_size
    actual_size="$(wc -c < "$output_path" | tr -d '[:space:]')"
    if [[ "$actual_size" != "$expected_size" ]]; then
      backup_path="${output_path}.bad.$(date +%Y%m%d%H%M%S)"
      echo "Size mismatch for ${output_path}: expected ${expected_size}, got ${actual_size}." >&2
      echo "Moving the bad file aside and retrying once from scratch:" >&2
      echo "  ${output_path} -> ${backup_path}" >&2
      mv "$output_path" "$backup_path"

      "$WGET_BIN" "${wget_args[@]}" \
        --output-document="$output_path" \
        "$url"

      actual_size="$(wc -c < "$output_path" | tr -d '[:space:]')"
      if [[ "$actual_size" != "$expected_size" ]]; then
        echo "Error: size mismatch after retry for ${output_path}: expected ${expected_size}, got ${actual_size}." >&2
        return 1
      fi
    fi
  fi
}

hf_repo_api_url() {
  local endpoint="$1"
  local repo="$2"
  local revision="$3"
  echo "${endpoint}/api/models/${repo}/revision/${revision}?blobs=true"
}

make_hf_manifest() {
  local endpoint="$1"
  local repo="$2"
  local revision="$3"
  local api_json="$4"
  local filter_mode="$5"

  "$PYTHON_BIN" - "$repo" "$revision" "$endpoint" "$api_json" "$filter_mode" <<'PY'
import json
import sys
from urllib.parse import quote

repo, revision, endpoint, api_json, filter_mode = sys.argv[1:]

with open(api_json, "r", encoding="utf-8") as f:
    payload = json.load(f)

siblings = payload.get("siblings")
if not siblings:
    message = payload.get("error") or payload.get("message") or "missing siblings in API response"
    raise SystemExit(f"Invalid Hugging Face API response for {repo}: {message}")

required_svor_files = {
    "remove_model_stage1.safetensors",
    "remove_model_stage2.safetensors",
}

seen = set()

for item in siblings:
    path = item.get("rfilename")
    if not path:
        continue
    if filter_mode == "svor-loras" and path not in required_svor_files:
        continue

    seen.add(path)
    size = item.get("size")
    encoded_path = quote(path, safe="/")
    encoded_revision = quote(revision, safe="")
    url = f"{endpoint}/{repo}/resolve/{encoded_revision}/{encoded_path}?download=true"
    print(f"{path}\t{size if size is not None else ''}\t{url}")

if filter_mode == "svor-loras":
    missing = sorted(required_svor_files - seen)
    if missing:
        raise SystemExit(f"{repo} is missing required LoRA file(s): {', '.join(missing)}")
PY
}

download_hf_repo_files() {
  local repo="$1"
  local local_dir="$2"
  local filter_mode="$3"
  local endpoint="$4"
  local auth_scope="${5:-hf}"
  local api_json
  api_json="$(mktemp)"

  echo "Fetching Hugging Face file list: ${repo}@${HF_REVISION} from ${endpoint}"
  local wget_args=("${WGET_COMMON_ARGS[@]}")
  if [[ "$auth_scope" == "hf" && -n "${HF_TOKEN:-}" ]]; then
    wget_args+=("--header=Authorization: Bearer ${HF_TOKEN}")
  fi
  "$WGET_BIN" "${wget_args[@]}" \
    --output-document="$api_json" \
    "$(hf_repo_api_url "$endpoint" "$repo" "$HF_REVISION")"

  local rel_path expected_size url
  while IFS=$'\t' read -r rel_path expected_size url; do
    [[ -n "$rel_path" ]] || continue
    download_file "$url" "${local_dir}/${rel_path}" "$expected_size" "$auth_scope"
  done < <(make_hf_manifest "$endpoint" "$repo" "$HF_REVISION" "$api_json" "$filter_mode")

  rm -f "$api_json"
}

download_hf_repo_files "$SVOR_REPO" "$MODEL_DIR" "svor-loras" "$HF_ENDPOINT" "hf"

echo
echo "Done. Expected layout:"
cat <<EOF

  ${MODEL_DIR}/remove_model_stage1.safetensors
  ${MODEL_DIR}/remove_model_stage2.safetensors

Tip: if Hugging Face returns 429/403, set HF_TOKEN and re-run:
  HF_TOKEN=hf_xxx bash download_weights_wget.sh ${MODEL_DIR}
EOF
