#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECTS=(project1 project2 project3)

# Determine identity source: file argument or AGE_SECRET_KEY env var
if [[ $# -ge 1 ]]; then
    IDENTITY_FILE="$1"
    decrypt() { age -d -i "$IDENTITY_FILE" -o "$2" "$1"; }
elif [[ -n "${AGE_SECRET_KEY:-}" ]]; then
    decrypt() { age -d -i - -o "$2" "$1" <<< "$AGE_SECRET_KEY"; }
else
    echo "Usage: $0 <path-to-private-key>"
    echo "   or: AGE_SECRET_KEY=... $0"
    exit 1
fi

for p in "${PROJECTS[@]}"; do
    input="${SCRIPT_DIR}/../${p}/.${p}_ci.age"
    output="${SCRIPT_DIR}/../${p}/${p}_ci_solutions.jl"

    if [[ ! -f "$input" ]]; then
        echo "Warning: $input not found, skipping"
        continue
    fi

    decrypt "$input" "$output"
    echo "Decrypted: $input -> $output"
done

echo "Done."
