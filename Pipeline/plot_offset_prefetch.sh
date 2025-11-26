#!/bin/bash
# Script to run plot_offset_prefetch_windows.py with default parameters
# Uses conda base environment python and sudo permissions

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use conda base python directly (works better with sudo)
# Priority: miniconda3 > anaconda3 > system python3
if [ -f "$HOME/miniconda3/bin/python" ]; then
    PYTHON_CMD="$HOME/miniconda3/bin/python"
    echo "Using conda base python: $PYTHON_CMD"
elif [ -f "$HOME/anaconda3/bin/python" ]; then
    PYTHON_CMD="$HOME/anaconda3/bin/python"
    echo "Using conda base python: $PYTHON_CMD"
else
    PYTHON_CMD="python3"
    echo "Warning: Conda base environment not found, using system python3"
fi

# Set default paths
ROOT_DIR="${SCRIPT_DIR}/prefetch_results/raw_results/Offset_results"
OUTPUT_DIR="${SCRIPT_DIR}/prefetch_results/plots"
BINS=20

# Check if root directory exists
if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Root directory not found: $ROOT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist (with sudo if needed)
if [ ! -w "$OUTPUT_DIR" ] 2>/dev/null; then
    sudo mkdir -p "$OUTPUT_DIR"
    # Set ownership to current user if created with sudo
    sudo chown -R "$USER:$USER" "$OUTPUT_DIR" 2>/dev/null || true
else
    mkdir -p "$OUTPUT_DIR"
fi

# Run the Python script with sudo
echo "Running plot_offset_prefetch_windows.py..."
echo "  Python: $PYTHON_CMD"
echo "  Root directory: $ROOT_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Bins: $BINS"
echo ""

sudo "$PYTHON_CMD" "${SCRIPT_DIR}/plot_offset_prefetch_windows.py" \
    --root "$ROOT_DIR" \
    --output "$OUTPUT_DIR" \
    --bins "$BINS"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "Success! Plots saved to: $OUTPUT_DIR"
    # Fix ownership of generated files
    sudo chown -R "$USER:$USER" "$OUTPUT_DIR" 2>/dev/null || true
    echo "Generated files:"
    ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
else
    echo ""
    echo "Error: Script execution failed."
    exit 1
fi

