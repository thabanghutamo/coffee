#!/bin/bash

echo "=== Vocal MIDI Generator Test Suite ==="

echo -e "\n1. Build Verification..."
ls build/VocalMIDI_artefacts/Standalone/VocalMIDI >/dev/null 2>&1 && echo "✓ Standalone built" || echo "✗ Standalone missing"
ls -d build/VocalMIDI_artefacts/VST3/VocalMIDI.vst3 >/dev/null 2>&1 && echo "✓ VST3 built" || echo "✗ VST3 missing"

echo -e "\n2. Model Verification..."
MODEL_COUNT=$(ls build/VocalMIDI_artefacts/models/*.onnx 2>/dev/null | wc -l)
[ "$MODEL_COUNT" -eq 8 ] && echo "✓ All 8 models present ($MODEL_COUNT/8)" || echo "✗ Only $MODEL_COUNT/8 models found"

echo -e "\n3. ONNX Runtime..."
export LD_LIBRARY_PATH=/workspaces/coffee/external/onnxruntime/lib:$LD_LIBRARY_PATH
ldd build/VocalMIDI_artefacts/Standalone/VocalMIDI 2>/dev/null | grep -q onnxruntime && echo "✓ ONNX Runtime linked" || echo "✗ ONNX Runtime not found"

echo -e "\n4. Python Dependencies..."
python3 -c "import torch; import onnxruntime; import numpy" 2>/dev/null && echo "✓ Python packages OK" || echo "✗ Missing packages"

echo -e "\n5. Model Inference Test..."
python3 << 'EOF'
import onnxruntime as ort
import numpy as np
try:
    session = ort.InferenceSession('build/VocalMIDI_artefacts/models/pitch_model.onnx')
    input_data = np.random.randn(1, 100, 128).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: input_data})
    print("✓ Model inference working")
except Exception as e:
    print(f"✗ Model inference failed: {e}")
EOF

echo -e "\n=== Test Suite Complete ==="
