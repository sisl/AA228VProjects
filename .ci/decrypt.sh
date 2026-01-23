#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <path-to-private-key>"
    exit 1
fi

IDENTITY_FILE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECTS=(project1 project2 project3)

for p in "${PROJECTS[@]}"; do
    input="${SCRIPT_DIR}/../${p}/.${p}_ci.age"
    output="${SCRIPT_DIR}/../${p}/${p}_ci_solutions.jl"

    if [[ ! -f "$input" ]]; then
        echo "Warning: $input not found, skipping"
        continue
    fi

    age -d -i "$IDENTITY_FILE" -o "$output" "$input"
    echo "Decrypted: $input -> $output"
done

echo "Done."
