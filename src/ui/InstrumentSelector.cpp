#include "InstrumentSelector.h"

InstrumentSelector::InstrumentSelector(VocalMIDIAudioProcessor& processor)
    : audioProcessor(processor)
{
    titleLabel.setText("Instruments", juce::dontSendNotification);
    titleLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel);
    
    // Instrument list would be populated here
    addAndMakeVisible(instrumentList);
}

InstrumentSelector::~InstrumentSelector()
{
}

void InstrumentSelector::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xFF2A2A2A));
}

void InstrumentSelector::resized()
{
    auto area = getLocalBounds().reduced(5);
    
    titleLabel.setBounds(area.removeFromTop(30));
    instrumentList.setBounds(area);
}
