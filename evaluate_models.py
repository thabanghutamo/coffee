#!/usr/bin/env python3
"""
Comprehensive model evaluation for Vocal MIDI Generator
Tests all 8 models with various inputs and measures performance
"""

import onnxruntime as ort
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple

def test_model_inference(model_path: Path, input_shape: Tuple[int, int, int]) -> Dict:
    """Test a single model's inference"""
    results = {
        'model': model_path.name,
        'size_kb': model_path.stat().st_size / 1024,
        'load_time_ms': 0,
        'inference_time_ms': 0,
        'success': False,
        'input_shape': input_shape,
        'output_shape': None,
        'output_range': None
    }
    
    try:
        # Test model loading
        start = time.time()
        session = ort.InferenceSession(str(model_path))
        results['load_time_ms'] = (time.time() - start) * 1000
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Test inference with synthetic data
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warm-up run
        session.run([output_name], {input_name: test_input})
        
        # Timed inference
        start = time.time()
        output = session.run([output_name], {input_name: test_input})[0]
        results['inference_time_ms'] = (time.time() - start) * 1000
        
        results['output_shape'] = output.shape
        results['output_range'] = (float(output.min()), float(output.max()))
        results['success'] = True
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def evaluate_all_models(models_dir: str = 'build/VocalMIDI_artefacts/models'):
    """Evaluate all models"""
    models_path = Path(models_dir)
    
    print("=" * 70)
    print("Vocal MIDI Generator - Model Evaluation")
    print("=" * 70)
    
    # Expected models
    model_files = [
        'pitch_model.onnx',
        'context_model.onnx',
        'timbre_model.onnx',
        'drum_generator.onnx',
        'bass_generator.onnx',
        'chord_generator.onnx',
        'melody_generator.onnx',
        'continuation_model.onnx'
    ]
    
    # Test configuration
    batch_size = 1
    seq_length = 100
    mel_bands = 128
    input_shape = (batch_size, seq_length, mel_bands)
    
    print(f"\nTest Configuration:")
    print(f"  Models directory: {models_path}")
    print(f"  Input shape: {input_shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    
    # Run tests
    results = []
    print(f"\n{'Model':<25} {'Size':<10} {'Load':<10} {'Infer':<10} {'Status':<10}")
    print("-" * 70)
    
    for model_file in model_files:
        model_path = models_path / model_file
        
        if not model_path.exists():
            print(f"{model_file:<25} {'MISSING':<10} {'---':<10} {'---':<10} {'✗ FAIL':<10}")
            continue
        
        result = test_model_inference(model_path, input_shape)
        results.append(result)
        
        if result['success']:
            print(f"{result['model']:<25} "
                  f"{result['size_kb']:>7.1f} KB "
                  f"{result['load_time_ms']:>7.1f} ms "
                  f"{result['inference_time_ms']:>7.1f} ms "
                  f"{'✓ PASS':<10}")
        else:
            print(f"{result['model']:<25} "
                  f"{result['size_kb']:>7.1f} KB "
                  f"{'ERROR':<10} {'---':<10} "
                  f"{'✗ FAIL':<10}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    
    successful = [r for r in results if r['success']]
    
    if successful:
        total_size = sum(r['size_kb'] for r in successful)
        avg_load = sum(r['load_time_ms'] for r in successful) / len(successful)
        avg_infer = sum(r['inference_time_ms'] for r in successful) / len(successful)
        max_infer = max(r['inference_time_ms'] for r in successful)
        
        print(f"\nModels tested: {len(results)}")
        print(f"Successful: {len(successful)}/{len(results)}")
        print(f"Total size: {total_size:.1f} KB ({total_size/1024:.2f} MB)")
        print(f"Average load time: {avg_load:.2f} ms")
        print(f"Average inference time: {avg_infer:.2f} ms")
        print(f"Max inference time: {max_infer:.2f} ms")
        
        # Performance assessment
        print("\nPerformance Assessment:")
        
        if avg_infer < 10:
            print("  ✓ Inference speed: EXCELLENT (< 10ms)")
        elif avg_infer < 20:
            print("  ✓ Inference speed: GOOD (< 20ms)")
        elif avg_infer < 50:
            print("  ⚠ Inference speed: ACCEPTABLE (< 50ms)")
        else:
            print("  ✗ Inference speed: TOO SLOW (> 50ms)")
        
        if total_size < 10240:  # 10 MB
            print("  ✓ Model size: COMPACT (< 10 MB)")
        elif total_size < 51200:  # 50 MB
            print("  ✓ Model size: REASONABLE (< 50 MB)")
        else:
            print("  ⚠ Model size: LARGE (> 50 MB)")
        
        # Latency budget
        print("\nReal-time Performance (100 frame buffer):")
        total_latency = avg_load + (avg_infer * 8)  # All 8 models
        print(f"  Estimated total latency: {total_latency:.2f} ms")
        
        if total_latency < 20:
            print("  ✓ Real-time capable (< 20ms target)")
        else:
            print("  ⚠ May not meet real-time requirements")
    
    # Detailed output
    print("\n" + "=" * 70)
    print("Detailed Results")
    print("=" * 70)
    
    for result in successful:
        print(f"\n{result['model']}:")
        print(f"  Input shape: {result['input_shape']}")
        print(f"  Output shape: {result['output_shape']}")
        print(f"  Output range: [{result['output_range'][0]:.3f}, {result['output_range'][1]:.3f}]")
        print(f"  File size: {result['size_kb']:.1f} KB")
        print(f"  Load time: {result['load_time_ms']:.2f} ms")
        print(f"  Inference time: {result['inference_time_ms']:.2f} ms")
    
    # Overall status
    print("\n" + "=" * 70)
    if len(successful) == len(model_files):
        print("✅ ALL MODELS PASSING - READY FOR PRODUCTION")
    else:
        print(f"⚠ {len(model_files) - len(successful)} MODELS FAILED")
    print("=" * 70)
    
    return results

if __name__ == '__main__':
    import sys
    
    models_dir = sys.argv[1] if len(sys.argv) > 1 else 'build/VocalMIDI_artefacts/models'
    results = evaluate_all_models(models_dir)
    
    # Exit code
    success_count = sum(1 for r in results if r['success'])
    sys.exit(0 if success_count == 8 else 1)
