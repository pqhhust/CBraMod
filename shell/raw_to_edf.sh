#!/bin/bash
# filepath: /home/user01/aiotlab/pqhung/CBraMod/shell/raw_to_edf.sh

# Find and convert all .eeg files recursively
find /home/user01/aiotlab/pqhung/EEG/EEG2100 -type f \( -name "*.eeg" -o -name "*.EEG" \) -exec bash -c '
    for file; do
        echo "Converting: $file"
        /home/user01/aiotlab/pqhung/EEG/nk2edf_ver15_source/nk2edf "$file"
        if [ $? -eq 0 ]; then
            echo "  ✓ Success: $(basename "$file")"
        else
            echo "  ✗ Error converting: $(basename "$file")"
        fi
    done
' bash {} +

echo "Conversion completed!"