#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include "PluginProcessor.h"

class InstrumentSelector : public juce::Component
{
public:
    InstrumentSelector(VocalMIDIAudioProcessor& processor);
    ~InstrumentSelector() override;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    
private:
    VocalMIDIAudioProcessor& audioProcessor;
    
    juce::Label titleLabel;
    juce::ListBox instrumentList;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(InstrumentSelector)
};
