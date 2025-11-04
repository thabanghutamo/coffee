#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <onnxruntime_cxx_api.h>
#include <vector>

/**
 * Utility class for converting between JUCE audio buffers and ONNX tensors
 * Optimized for real-time audio processing with minimal allocations
 */
class TensorConverter
{
public:
    TensorConverter();
    ~TensorConverter();
    
    /**
     * Convert JUCE audio buffer to ONNX tensor
     * @param audioBuffer Source audio buffer
     * @param numSamples Number of samples to convert
     * @param numChannels Number of channels (1 for mono, 2 for stereo)
     * @return ONNX tensor ready for model inference
     */
    Ort::Value audioBufferToTensor(
        const float* const* audioBuffer,
        int numSamples,
        int numChannels
    );
    
    /**
     * Convert mel-spectrogram to ONNX tensor
     * @param melSpec 2D vector [time][mel_bands]
     * @param batchSize Batch dimension (usually 1 for real-time)
     * @return ONNX tensor with shape [batch, time, mel_bands]
     */
    Ort::Value melSpectrogramToTensor(
        const std::vector<std::vector<float>>& melSpec,
        int batchSize = 1
    );
    
    /**
     * Convert feature vector to ONNX tensor
     * @param features 1D feature vector
     * @param shape Desired tensor shape
     * @return ONNX tensor
     */
    Ort::Value featureVectorToTensor(
        const std::vector<float>& features,
        const std::vector<int64_t>& shape
    );
    
    /**
     * Extract float array from ONNX tensor output
     * @param tensor ONNX output tensor
     * @return Vector of floats
     */
    std::vector<float> tensorToVector(Ort::Value& tensor);
    
    /**
     * Extract 2D array from ONNX tensor output
     * @param tensor ONNX output tensor with shape [time, features]
     * @return 2D vector
     */
    std::vector<std::vector<float>> tensorTo2DVector(Ort::Value& tensor);
    
    /**
     * Convert pitch predictions to MIDI note events
     * @param pitchTensor Tensor with shape [time, 128] containing note probabilities
     * @param threshold Probability threshold for note activation
     * @return Vector of MIDI messages
     */
    juce::MidiBuffer tensorToMIDI(
        Ort::Value& pitchTensor,
        float threshold = 0.5f,
        int sampleRate = 44100
    );
    
private:
    std::unique_ptr<Ort::MemoryInfo> memoryInfo;
    
    // Reusable buffers to avoid allocations
    std::vector<float> tempBuffer;
    std::vector<int64_t> tempShape;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TensorConverter)
};
