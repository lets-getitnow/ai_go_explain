#!/bin/bash
# Example: Running the Human Games Pipeline
# ========================================
# This script shows how to run the complete pipeline on human SGF games

set -e  # Exit on any error

echo "🚀 Starting Human Games Analysis Pipeline"
echo "=========================================="

# Configuration
INPUT_DIR="games/go13"
OUTPUT_DIR="human_games_analysis"
MODEL_PATH="models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt"

# Check prerequisites
echo "📋 Checking prerequisites..."

if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Input directory $INPUT_DIR does not exist"
    echo "Please add SGF files to $INPUT_DIR"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file $MODEL_PATH does not exist"
    echo "Please download a KataGo model checkpoint"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Run the complete pipeline
echo "🔧 Running complete pipeline..."
python run_human_games_pipeline.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --model-path "$MODEL_PATH"

echo ""
echo "🎉 Pipeline completed successfully!"
echo ""
echo "📊 Results available in: $OUTPUT_DIR"
echo "🌐 HTML reports: $OUTPUT_DIR/html_reports"
echo ""
echo "To view the results, open:"
echo "  $OUTPUT_DIR/html_reports/index.html"
echo ""
echo "Or run:"
echo "  open $OUTPUT_DIR/html_reports/index.html" 