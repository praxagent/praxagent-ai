#!/usr/bin/env bash
# Double-fork + setsid so the serve supervisor is not killed when the
# launching make/Cursor shell exits (same process-group cleanup).
set -eu

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

log_file="${1:?}"
pid_file="${2:?}"
shift 2

exec python3 - "$root" "$log_file" "$pid_file" "$@" <<'PY'
import os, sys

root, log_file, pid_file, *cmd = sys.argv[1:]
supervise = os.path.join(root, "scripts", "supervise_serve.sh")
os.chdir(root)

# First fork: return control to make immediately.
if os.fork() > 0:
    sys.exit(0)

os.setsid()

# Second fork: ensure we can never re-acquire a controlling terminal.
if os.fork() > 0:
    sys.exit(0)

devnull = os.open("/dev/null", os.O_RDWR)
os.dup2(devnull, 0)
os.dup2(devnull, 1)
os.dup2(devnull, 2)
if devnull > 2:
    os.close(devnull)

os.execv(supervise, [supervise, log_file, pid_file, *cmd])
PY
