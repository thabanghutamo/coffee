#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>
#include <string>
#include <memory>

// ONNX Runtime headers
#include <onnxruntime_cxx_api.h>

struct GeneratedMIDI
{
    std::vector<std::vector<float>> pitch;      // [time][128] pitch probabilities
    std::vector<std::vector<float>> velocity;   // [time][128] velocities
    std::vector<std::vector<float>> timing;     // [time][timing_bins]
    std::vector<float> onset;                    // [time] onset probabilities
    std::vector<float> offset;                   // [time] offset probabilities
    
    juce::String genre;
    juce::String instrument;
    float bpm;
};

class ModelInference
{
public:
    ModelInference();
    ~ModelInference();
    
    // Load ONNX models from disk
    bool loadModels();
    bool loadModel(const juce::String& modelName, const juce::File& modelFile, std::unique_ptr<Ort::Session>* session);
    
    // Generate MIDI from audio features
    GeneratedMIDI generateMIDI(
        const std::vector<std::vector<float>>& melSpectrogram,
        const struct PitchInfo& pitchInfo,
        const struct RhythmInfo& rhythmInfo,
        const juce::String& genre,
        float bpm,
        const juce::String& mood,
        const juce::String& primaryInstrument
    );
    
    // Generate multi-track MIDI
    std::vector<GeneratedMIDI> generateMultiTrack(
        const std::vector<std::vector<float>>& melSpectrogram,
        const juce::String& genre,
        float bpm
    );
    
    // Regenerate specific track
    GeneratedMIDI regenerateTrack(
        const juce::String& trackType,
        const std::vector<std::vector<float>>& contextFeatures
    );
    
    bool isLoaded() const { return modelsLoaded; }
    
private:
    // ONNX Runtime environment
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;
    
    // Model sessions
    std::unique_ptr<Ort::Session> pitchSession;
    std::unique_ptr<Ort::Session> contextSession;
    std::unique_ptr<Ort::Session> timbreSession;
    std::unique_ptr<Ort::Session> drumSession;
    std::unique_ptr<Ort::Session> bassSession;
    std::unique_ptr<Ort::Session> chordSession;
    std::unique_ptr<Ort::Session> melodySession;
    std::unique_ptr<Ort::Session> continuationSession;
    
    bool modelsLoaded;
    juce::File modelsDirectory;
    
    // Helper methods
    std::vector<float> prepareInput(const std::vector<std::vector<float>>& features);
    void runInference(Ort::Session* session, const std::vector<float>& input, std::vector<float>& output);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ModelInference)
};
