#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include "audio/AudioCapture.h"
#include "audio/FeatureExtraction.h"
#include "audio/PitchDetector.h"
#include "audio/RhythmAnalyzer.h"
#include "ml/ModelInference.h"
#include "midi/MIDIGenerator.h"
#include "midi/MIDITrack.h"
#include <vector>
#include <memory>

class VocalMIDIAudioProcessor : public juce::AudioProcessor
{
public:
    VocalMIDIAudioProcessor();
    ~VocalMIDIAudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

#ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
#endif

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // Parameter accessors
    juce::AudioProcessorValueTreeState& getParameters() { return parameters; }
    
    // User settings
    void setGenre(const juce::String& genre);
    void setBPM(float bpm);
    void setMood(const juce::String& mood);
    void setPrimaryInstrument(const juce::String& instrument);
    
    // Track management
    std::vector<std::shared_ptr<MIDITrack>>& getTracks() { return midiTracks; }
    void regenerateTrack(int trackIndex);
    void muteTrack(int trackIndex, bool mute);
    void soloTrack(int trackIndex, bool solo);
    
    // MIDI export
    void exportMIDI(const juce::File& outputFile);
    
private:
    // Audio processing components
    std::unique_ptr<AudioCapture> audioCapture;
    std::unique_ptr<FeatureExtraction> featureExtractor;
    std::unique_ptr<PitchDetector> pitchDetector;
    std::unique_ptr<RhythmAnalyzer> rhythmAnalyzer;
    
    // ML inference engines
    std::unique_ptr<ModelInference> modelInference;
    
    // MIDI generation
    std::unique_ptr<MIDIGenerator> midiGenerator;
    std::vector<std::shared_ptr<MIDITrack>> midiTracks;
    
    // Parameters
    juce::AudioProcessorValueTreeState parameters;
    
    // User settings
    juce::String currentGenre;
    float currentBPM;
    juce::String currentMood;
    juce::String primaryInstrument;
    
    // Processing state
    double sampleRate;
    int blockSize;
    bool isProcessing;
    
    // Circular buffer for audio input
    juce::AudioBuffer<float> audioBuffer;
    int writePosition;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VocalMIDIAudioProcessor)
};
