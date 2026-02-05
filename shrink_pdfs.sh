#!/bin/bash

TARGET_KB=2048

find . -type f -iname "*.pdf" ! -name "*_compressed.pdf" -print0 |
while IFS= read -r -d '' file; do

    base_name="${file%.*}"
    out_file="${base_name}_compressed.pdf"

    if [[ -f "$out_file" ]]; then
        echo "⏭️  Skipping: $out_file already exists."
        continue
    fi

    # File size in bytes
    actual_size=$(stat -c%s "$file")
    actual_kb=$((actual_size / 1024))
    actual_mb=$((actual_kb / 1024))

    if (( actual_kb <= TARGET_KB )); then
        echo "✅ Already under 2MB: $file (${actual_mb} MB). Skipping."
        continue
    fi

    echo "🚀 Compressing: $file (${actual_mb} MB)"

    gs -sDEVICE=pdfwrite \
       -dCompatibilityLevel=1.4 \
       -dPDFSETTINGS=/ebook \
       -dNOPAUSE -dQUIET -dBATCH \
       -sOutputFile="$out_file" \
       "$file"

    if [[ -f "$out_file" ]]; then

        new_size=$(stat -c%s "$out_file")
        new_kb=$((new_size / 1024))

        if (( new_kb > TARGET_KB )); then
            echo "⚠️  Still over 2MB ($new_kb KB). Retrying with max compression..."

            gs -sDEVICE=pdfwrite \
               -dCompatibilityLevel=1.4 \
               -dPDFSETTINGS=/screen \
               -dNOPAUSE -dQUIET -dBATCH \
               -sOutputFile="$out_file" \
               "$file"

            new_size=$(stat -c%s "$out_file")
            new_kb=$((new_size / 1024))
        fi

        echo "✨ Done: $out_file ($new_kb KB)"
    else
        echo "❌ Error: Could not process $file."
    fi

    echo "--------------------------------"

done
