#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include "PluginProcessor.h"
#include "ui/PianoRollComponent.h"
#include "ui/ControlPanel.h"
#include "ui/InstrumentSelector.h"

class VocalMIDIAudioProcessorEditor : public juce::AudioProcessorEditor,
                                       private juce::Timer
{
public:
    VocalMIDIAudioProcessorEditor(VocalMIDIAudioProcessor&);
    ~VocalMIDIAudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    
private:
    void timerCallback() override;
    
    VocalMIDIAudioProcessor& audioProcessor;
    
    // UI Components
    std::unique_ptr<PianoRollComponent> pianoRoll;
    std::unique_ptr<ControlPanel> controlPanel;
    std::unique_ptr<InstrumentSelector> instrumentSelector;
    
    // Layout
    juce::Rectangle<int> headerArea;
    juce::Rectangle<int> pianoRollArea;
    juce::Rectangle<int> controlArea;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VocalMIDIAudioProcessorEditor)
};
