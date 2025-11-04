#!/bin/bash

# Vocal MIDI Generator - Standalone Launch Script

export LD_LIBRARY_PATH=/workspaces/coffee/external/onnxruntime/lib:$LD_LIBRARY_PATH

echo "========================================="
echo "Vocal MIDI Generator - Standalone Plugin"
echo "========================================="
echo ""
echo "ONNX Runtime: $(ls /workspaces/coffee/external/onnxruntime/lib/libonnxruntime.so.* 2>/dev/null | head -1)"
echo "Models: $(ls /workspaces/coffee/build/VocalMIDI_artefacts/models/*.onnx 2>/dev/null | wc -l) ONNX files loaded"
echo ""
echo "Starting plugin..."
echo ""

cd /workspaces/coffee/build/VocalMIDI_artefacts/Standalone
./VocalMIDI "$@"
