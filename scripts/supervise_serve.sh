#!/usr/bin/env bash
# Supervise a long-running blog/site preview process.
# Restarts on unexpected exit; stops cleanly on SIGTERM/SIGINT.
#
# Do NOT trap HUP here: nohup ignores HUP so the process survives the
# launching make/shell exiting. Trapping HUP would undo that and kill the
# server every time `make ... up` returns.
set -eu

log_file="${1:?log file required}"
shift
pid_file="${1:?pid file required}"
shift

child=""
stopping=0

term() {
  stopping=1
  if [[ -n "${child}" ]] && kill -0 "${child}" 2>/dev/null; then
    kill "${child}" 2>/dev/null || true
    wait "${child}" 2>/dev/null || true
  fi
  rm -f "${pid_file}" "${pid_file}.child"
  exit 0
}

trap term TERM INT
# Ignore hangup so we outlive the make recipe shell even without nohup.
trap '' HUP

mkdir -p "$(dirname "${log_file}")" "$(dirname "${pid_file}")"
echo "$$" > "${pid_file}"

while true; do
  "$@" >>"${log_file}" 2>&1 &
  child=$!
  # Expose the real server pid for tooling that checks the listener.
  echo "${child}" > "${pid_file}.child"
  set +e
  wait "${child}"
  code=$?
  set -e
  rm -f "${pid_file}.child"
  if [[ "${stopping}" -eq 1 ]]; then
    exit 0
  fi
  {
    echo ""
    echo "[supervise] process exited (code ${code}) at $(date '+%Y-%m-%d %H:%M:%S'); restarting in 1s"
    echo ""
  } >>"${log_file}"
  sleep 1
done
