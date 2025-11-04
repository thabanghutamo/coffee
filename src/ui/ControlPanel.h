#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include "PluginProcessor.h"

class ControlPanel : public juce::Component
{
public:
    ControlPanel(VocalMIDIAudioProcessor& processor);
    ~ControlPanel() override;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    
private:
    VocalMIDIAudioProcessor& audioProcessor;
    
    // Genre selection
    juce::Label genreLabel;
    juce::ComboBox genreSelector;
    
    // BPM control
    juce::Label bpmLabel;
    juce::Slider bpmSlider;
    
    // Mood selection
    juce::Label moodLabel;
    juce::ComboBox moodSelector;
    
    // Voice controls
    juce::ToggleButton autoBendButton;
    juce::ToggleButton autoKeyButton;
    
    // Sensitivity
    juce::Label sensitivityLabel;
    juce::Slider sensitivitySlider;
    
    // Generate button
    juce::TextButton generateButton;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ControlPanel)
};
