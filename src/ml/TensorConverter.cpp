#include "TensorConverter.h"

TensorConverter::TensorConverter()
{
    memoryInfo = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    // Pre-allocate buffers for common sizes
    tempBuffer.reserve(128 * 1024); // Reserve 128k floats
    tempShape.reserve(4); // Most tensors are 2-4 dimensions
}

TensorConverter::~TensorConverter()
{
}

Ort::Value TensorConverter::audioBufferToTensor(
    const float* const* audioBuffer,
    int numSamples,
    int numChannels)
{
    // Reshape to [batch=1, channels, samples]
    tempShape = {1, static_cast<int64_t>(numChannels), static_cast<int64_t>(numSamples)};
    
    // Interleave audio data if stereo
    tempBuffer.resize(numChannels * numSamples);
    
    if (numChannels == 1)
    {
        // Mono: direct copy
        std::copy(audioBuffer[0], audioBuffer[0] + numSamples, tempBuffer.begin());
    }
    else
    {
        // Stereo/multi-channel: interleave
        for (int ch = 0; ch < numChannels; ++ch)
        {
            for (int s = 0; s < numSamples; ++s)
            {
                tempBuffer[ch * numSamples + s] = audioBuffer[ch][s];
            }
        }
    }
    
    return Ort::Value::CreateTensor<float>(
        *memoryInfo,
        tempBuffer.data(),
        tempBuffer.size(),
        tempShape.data(),
        tempShape.size()
    );
}

Ort::Value TensorConverter::melSpectrogramToTensor(
    const std::vector<std::vector<float>>& melSpec,
    int batchSize)
{
    if (melSpec.empty())
    {
        // Return empty tensor
        tempShape = {0, 0, 0};
        return Ort::Value::CreateTensor<float>(
            *memoryInfo,
            nullptr,
            0,
            tempShape.data(),
            tempShape.size()
        );
    }
    
    int numFrames = static_cast<int>(melSpec.size());
    int numBands = static_cast<int>(melSpec[0].size());
    
    // Shape: [batch, time, mel_bands]
    tempShape = {
        static_cast<int64_t>(batchSize),
        static_cast<int64_t>(numFrames),
        static_cast<int64_t>(numBands)
    };
    
    // Flatten to 1D array
    tempBuffer.resize(batchSize * numFrames * numBands);
    
    for (int b = 0; b < batchSize; ++b)
    {
        for (int t = 0; t < numFrames; ++t)
        {
            for (int mel = 0; mel < numBands; ++mel)
            {
                int idx = b * (numFrames * numBands) + t * numBands + mel;
                tempBuffer[idx] = melSpec[t][mel];
            }
        }
    }
    
    return Ort::Value::CreateTensor<float>(
        *memoryInfo,
        tempBuffer.data(),
        tempBuffer.size(),
        tempShape.data(),
        tempShape.size()
    );
}

Ort::Value TensorConverter::featureVectorToTensor(
    const std::vector<float>& features,
    const std::vector<int64_t>& shape)
{
    // Validate that shape matches feature count
    int64_t totalElements = 1;
    for (auto dim : shape)
    {
        totalElements *= dim;
    }
    
    if (totalElements != static_cast<int64_t>(features.size()))
    {
        DBG("Warning: Shape mismatch in featureVectorToTensor");
    }
    
    return Ort::Value::CreateTensor<float>(
        *memoryInfo,
        const_cast<float*>(features.data()),
        features.size(),
        shape.data(),
        shape.size()
    );
}

std::vector<float> TensorConverter::tensorToVector(Ort::Value& tensor)
{
    float* data = tensor.GetTensorMutableData<float>();
    auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    
    size_t totalElements = 1;
    for (auto dim : shape)
    {
        totalElements *= dim;
    }
    
    return std::vector<float>(data, data + totalElements);
}

std::vector<std::vector<float>> TensorConverter::tensorTo2DVector(Ort::Value& tensor)
{
    float* data = tensor.GetTensorMutableData<float>();
    auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    
    if (shape.size() < 2)
    {
        DBG("Warning: Tensor is not 2D");
        return {};
    }
    
    int rows = static_cast<int>(shape[0]);
    int cols = static_cast<int>(shape[1]);
    
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
    
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            result[r][c] = data[r * cols + c];
        }
    }
    
    return result;
}

juce::MidiBuffer TensorConverter::tensorToMIDI(
    Ort::Value& pitchTensor,
    float threshold,
    int sampleRate)
{
    juce::MidiBuffer midiBuffer;
    
    float* data = pitchTensor.GetTensorMutableData<float>();
    auto shape = pitchTensor.GetTensorTypeAndShapeInfo().GetShape();
    
    if (shape.size() < 2)
    {
        return midiBuffer;
    }
    
    int numFrames = static_cast<int>(shape[0]);
    int numNotes = static_cast<int>(shape[1]); // Should be 128
    
    // Track which notes are currently on
    std::array<bool, 128> noteStates;
    noteStates.fill(false);
    
    // Convert frame index to sample position (assuming 10ms hop size)
    const int hopSize = sampleRate / 100; // 10ms at 44.1kHz = 441 samples
    
    for (int frame = 0; frame < numFrames; ++frame)
    {
        int samplePos = frame * hopSize;
        
        for (int note = 0; note < std::min(numNotes, 128); ++note)
        {
            float probability = data[frame * numNotes + note];
            bool shouldBeOn = probability > threshold;
            
            if (shouldBeOn && !noteStates[note])
            {
                // Note on
                int velocity = static_cast<int>(std::min(127.0f, probability * 127.0f));
                midiBuffer.addEvent(juce::MidiMessage::noteOn(1, note, static_cast<juce::uint8>(velocity)), samplePos);
                noteStates[note] = true;
            }
            else if (!shouldBeOn && noteStates[note])
            {
                // Note off
                midiBuffer.addEvent(juce::MidiMessage::noteOff(1, note), samplePos);
                noteStates[note] = false;
            }
        }
    }
    
    // Turn off any remaining notes
    int finalPos = numFrames * hopSize;
    for (int note = 0; note < 128; ++note)
    {
        if (noteStates[note])
        {
            midiBuffer.addEvent(juce::MidiMessage::noteOff(1, note), finalPos);
        }
    }
    
    return midiBuffer;
}
