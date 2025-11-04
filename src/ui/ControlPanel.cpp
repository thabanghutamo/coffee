#include "ControlPanel.h"

ControlPanel::ControlPanel(VocalMIDIAudioProcessor& processor)
    : audioProcessor(processor)
{
    // Genre selector
    genreLabel.setText("Genre", juce::dontSendNotification);
    addAndMakeVisible(genreLabel);
    
    genreSelector.addItem("Trap", 1);
    genreSelector.addItem("Pop", 2);
    genreSelector.addItem("Afrobeat", 3);
    genreSelector.addItem("Hip Hop", 4);
    genreSelector.addItem("R&B", 5);
    genreSelector.addItem("Electronic", 6);
    genreSelector.setSelectedId(1);
    addAndMakeVisible(genreSelector);
    
    // BPM slider
    bpmLabel.setText("BPM", juce::dontSendNotification);
    addAndMakeVisible(bpmLabel);
    
    bpmSlider.setRange(60.0, 200.0, 1.0);
    bpmSlider.setValue(120.0);
    bpmSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    addAndMakeVisible(bpmSlider);
    
    // Mood selector
    moodLabel.setText("Mood", juce::dontSendNotification);
    addAndMakeVisible(moodLabel);
    
    moodSelector.addItem("Dark", 1);
    moodSelector.addItem("Chill", 2);
    moodSelector.addItem("Upbeat", 3);
    moodSelector.addItem("Cinematic", 4);
    moodSelector.addItem("Energetic", 5);
    moodSelector.setSelectedId(3);
    addAndMakeVisible(moodSelector);
    
    // Voice controls
    autoBendButton.setButtonText("Auto Bend");
    autoBendButton.setToggleState(true, juce::dontSendNotification);
    addAndMakeVisible(autoBendButton);
    
    autoKeyButton.setButtonText("Auto Key");
    autoKeyButton.setToggleState(true, juce::dontSendNotification);
    addAndMakeVisible(autoKeyButton);
    
    // Sensitivity
    sensitivityLabel.setText("Sensitivity", juce::dontSendNotification);
    addAndMakeVisible(sensitivityLabel);
    
    sensitivitySlider.setRange(0.0, 1.0, 0.01);
    sensitivitySlider.setValue(0.7);
    sensitivitySlider.setSliderStyle(juce::Slider::LinearVertical);
    sensitivitySlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 20);
    addAndMakeVisible(sensitivitySlider);
    
    // Generate button
    generateButton.setButtonText("Generate MIDI");
    addAndMakeVisible(generateButton);
}

ControlPanel::~ControlPanel()
{
}

void ControlPanel::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xFF252525));
    
    g.setColour(juce::Colours::white);
    g.setFont(16.0f);
    g.drawText("Controls", getLocalBounds().removeFromTop(40),
               juce::Justification::centred, true);
}

void ControlPanel::resized()
{
    auto area = getLocalBounds().reduced(10);
    area.removeFromTop(40); // Title space
    
    auto row = area.removeFromTop(60);
    genreLabel.setBounds(row.removeFromTop(20));
    genreSelector.setBounds(row.removeFromTop(30));
    
    area.removeFromTop(10);
    row = area.removeFromTop(60);
    bpmLabel.setBounds(row.removeFromTop(20));
    bpmSlider.setBounds(row.removeFromTop(30));
    
    area.removeFromTop(10);
    row = area.removeFromTop(60);
    moodLabel.setBounds(row.removeFromTop(20));
    moodSelector.setBounds(row.removeFromTop(30));
    
    area.removeFromTop(20);
    autoBendButton.setBounds(area.removeFromTop(30));
    area.removeFromTop(10);
    autoKeyButton.setBounds(area.removeFromTop(30));
    
    area.removeFromTop(20);
    sensitivityLabel.setBounds(area.removeFromTop(20));
    sensitivitySlider.setBounds(area.removeFromTop(100));
    
    area.removeFromTop(20);
    generateButton.setBounds(area.removeFromTop(40));
}
