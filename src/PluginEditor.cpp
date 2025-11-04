#include "PluginEditor.h"

VocalMIDIAudioProcessorEditor::VocalMIDIAudioProcessorEditor(VocalMIDIAudioProcessor& p)
    : AudioProcessorEditor(&p), audioProcessor(p)
{
    setSize(1200, 800);
    
    // Create UI components
    pianoRoll = std::make_unique<PianoRollComponent>(audioProcessor);
    addAndMakeVisible(pianoRoll.get());
    
    controlPanel = std::make_unique<ControlPanel>(audioProcessor);
    addAndMakeVisible(controlPanel.get());
    
    instrumentSelector = std::make_unique<InstrumentSelector>(audioProcessor);
    addAndMakeVisible(instrumentSelector.get());
    
    // Start timer for UI updates (30 FPS)
    startTimer(33);
}

VocalMIDIAudioProcessorEditor::~VocalMIDIAudioProcessorEditor()
{
    stopTimer();
}

void VocalMIDIAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xFF1E1E1E));
    
    g.setColour(juce::Colours::white);
    g.setFont(24.0f);
    g.drawText("Vocal MIDI Generator", headerArea, juce::Justification::centred, true);
}

void VocalMIDIAudioProcessorEditor::resized()
{
    auto area = getLocalBounds();
    
    // Header
    headerArea = area.removeFromTop(60);
    
    // Control panel on the right
    controlArea = area.removeFromRight(300);
    controlPanel->setBounds(controlArea);
    
    // Instrument selector on the left
    auto selectorArea = area.removeFromLeft(200);
    instrumentSelector->setBounds(selectorArea);
    
    // Piano roll takes the rest
    pianoRollArea = area;
    pianoRoll->setBounds(pianoRollArea);
}

void VocalMIDIAudioProcessorEditor::timerCallback()
{
    // Update UI components
    pianoRoll->repaint();
}
