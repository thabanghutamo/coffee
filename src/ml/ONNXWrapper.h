#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>

class ONNXWrapper
{
public:
    ONNXWrapper();
    ~ONNXWrapper();
    
    bool loadModel(const juce::File& modelPath);
    bool isLoaded() const { return modelLoaded; }
    
    // Run inference
    std::vector<float> runInference(const std::vector<float>& input);
    
    // Batch inference
    std::vector<std::vector<float>> runBatchInference(
        const std::vector<std::vector<float>>& batchInput);
    
private:
    bool modelLoaded;
    juce::String modelPath;
    
    // ONNX Runtime session (forward declaration to avoid ONNX headers here)
    void* ortSession;
    void* ortEnv;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ONNXWrapper)
};
