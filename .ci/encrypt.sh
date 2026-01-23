#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPIENTS_FILE="${SCRIPT_DIR}/recipients.txt"
CI_PUBLIC_KEY="age19ufmu0gh58n0wznvl7jyjtgq94j84jqtms369tqv9pr9mpds4gzsfzq5kj"
PROJECTS=(project1 project2 project3)

# Build recipient file from GitHub handles + CI key
recipients_tmp=$(mktemp)
trap "rm -f $recipients_tmp" EXIT

echo "Fetching public keys from GitHub..."
while IFS= read -r handle || [[ -n "$handle" ]]; do
    [[ -z "$handle" || "$handle" =~ ^# ]] && continue
    echo "  - $handle"
    curl -sf "https://github.com/${handle}.keys" >> "$recipients_tmp"
done < "$RECIPIENTS_FILE"

echo "$CI_PUBLIC_KEY" >> "$recipients_tmp"
echo "Added CI runner public key"

# Encrypt each project's solution file
for p in "${PROJECTS[@]}"; do
    input="${SCRIPT_DIR}/../${p}/${p}_ci_solutions.jl"
    output="${SCRIPT_DIR}/../${p}/.${p}_ci.age"

    if [[ ! -f "$input" ]]; then
        echo "Warning: $input not found, skipping"
        continue
    fi

    age -R "$recipients_tmp" -o "$output" "$input"
    echo "Encrypted: $input -> $output"
done

echo "Done."
