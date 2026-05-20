#!/usr/bin/env bash
# Starts docker compose while keeping macOS awake (caffeinate).
# On Linux/Windows, caffeinate is not available and docker compose runs directly.
if command -v caffeinate &>/dev/null; then
  exec caffeinate -d -i -s docker compose up "$@"
else
  exec docker compose up "$@"
fi
