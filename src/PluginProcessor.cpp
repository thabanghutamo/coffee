#include "PluginProcessor.h"
#include "PluginEditor.h"

VocalMIDIAudioProcessor::VocalMIDIAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor(BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput("Input", juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       ),
#endif
      parameters(*this, nullptr, juce::Identifier("VocalMIDIParameters"),
                 {
                     std::make_unique<juce::AudioParameterFloat>("threshold", "Threshold", 0.0f, 1.0f, 0.3f),
                     std::make_unique<juce::AudioParameterFloat>("sensitivity", "Sensitivity", 0.0f, 1.0f, 0.7f),
                     std::make_unique<juce::AudioParameterFloat>("quantize", "Quantize", 0.0f, 1.0f, 0.5f),
                     std::make_unique<juce::AudioParameterBool>("autoBend", "Auto Bend", true),
                     std::make_unique<juce::AudioParameterBool>("autoKey", "Auto Key", true)
                 }),
      currentGenre("Trap"),
      currentBPM(120.0f),
      currentMood("Upbeat"),
      primaryInstrument("Piano"),
      sampleRate(44100.0),
      blockSize(512),
      isProcessing(false),
      writePosition(0)
{
    // Initialize audio processing components
    audioCapture = std::make_unique<AudioCapture>();
    featureExtractor = std::make_unique<FeatureExtraction>();
    pitchDetector = std::make_unique<PitchDetector>();
    rhythmAnalyzer = std::make_unique<RhythmAnalyzer>();
    
    // Initialize ML inference
    modelInference = std::make_unique<ModelInference>();
    
    // Initialize MIDI generator
    midiGenerator = std::make_unique<MIDIGenerator>();
    
    // Prepare audio buffer (10 seconds at 48kHz)
    audioBuffer.setSize(2, 48000 * 10);
}

VocalMIDIAudioProcessor::~VocalMIDIAudioProcessor()
{
}

const juce::String VocalMIDIAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool VocalMIDIAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool VocalMIDIAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool VocalMIDIAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double VocalMIDIAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int VocalMIDIAudioProcessor::getNumPrograms()
{
    return 1;
}

int VocalMIDIAudioProcessor::getCurrentProgram()
{
    return 0;
}

void VocalMIDIAudioProcessor::setCurrentProgram(int index)
{
}

const juce::String VocalMIDIAudioProcessor::getProgramName(int index)
{
    return {};
}

void VocalMIDIAudioProcessor::changeProgramName(int index, const juce::String& newName)
{
}

void VocalMIDIAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    this->sampleRate = sampleRate;
    this->blockSize = samplesPerBlock;
    
    // Initialize components with sample rate
    audioCapture->prepare(sampleRate, samplesPerBlock);
    featureExtractor->prepare(sampleRate);
    pitchDetector->prepare(sampleRate);
    rhythmAnalyzer->prepare(sampleRate, currentBPM);
    
    // Load ML models
    modelInference->loadModels();
}

void VocalMIDIAudioProcessor::releaseResources()
{
    audioCapture->reset();
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool VocalMIDIAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused(layouts);
    return true;
  #else
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
#endif

void VocalMIDIAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    
    // Capture audio input
    audioCapture->process(buffer);
    
    // Extract features from audio
    auto features = featureExtractor->extractMelSpectrogram(buffer);
    
    // Detect pitch and rhythm
    auto pitchInfo = pitchDetector->detectPitch(buffer);
    auto rhythmInfo = rhythmAnalyzer->analyzeRhythm(buffer);
    
    // Run ML inference to generate MIDI
    if (isProcessing && !features.melSpectrogram.empty())
    {
        auto generatedMIDI = modelInference->generateMIDI(
            features.melSpectrogram,
            pitchInfo,
            rhythmInfo,
            currentGenre,
            currentBPM,
            currentMood,
            primaryInstrument
        );
        
        // Update MIDI tracks
        midiGenerator->updateTracks(generatedMIDI, midiTracks);
        
        // Output MIDI to DAW
        for (auto& track : midiTracks)
        {
            if (!track->isMuted())
            {
                track->addToMidiBuffer(midiMessages);
            }
        }
    }
}

bool VocalMIDIAudioProcessor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor* VocalMIDIAudioProcessor::createEditor()
{
    return new VocalMIDIAudioProcessorEditor(*this);
}

void VocalMIDIAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = parameters.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void VocalMIDIAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    
    if (xmlState.get() != nullptr)
        if (xmlState->hasTagName(parameters.state.getType()))
            parameters.replaceState(juce::ValueTree::fromXml(*xmlState));
}

void VocalMIDIAudioProcessor::setGenre(const juce::String& genre)
{
    currentGenre = genre;
}

void VocalMIDIAudioProcessor::setBPM(float bpm)
{
    currentBPM = bpm;
    rhythmAnalyzer->setBPM(bpm);
}

void VocalMIDIAudioProcessor::setMood(const juce::String& mood)
{
    currentMood = mood;
}

void VocalMIDIAudioProcessor::setPrimaryInstrument(const juce::String& instrument)
{
    primaryInstrument = instrument;
}

void VocalMIDIAudioProcessor::regenerateTrack(int trackIndex)
{
    if (trackIndex >= 0 && trackIndex < midiTracks.size())
    {
        // Request regeneration from ML model
        midiGenerator->regenerateTrack(midiTracks[trackIndex], modelInference.get());
    }
}

void VocalMIDIAudioProcessor::muteTrack(int trackIndex, bool mute)
{
    if (trackIndex >= 0 && trackIndex < midiTracks.size())
    {
        midiTracks[trackIndex]->setMuted(mute);
    }
}

void VocalMIDIAudioProcessor::soloTrack(int trackIndex, bool solo)
{
    if (trackIndex >= 0 && trackIndex < midiTracks.size())
    {
        midiTracks[trackIndex]->setSolo(solo);
    }
}

void VocalMIDIAudioProcessor::exportMIDI(const juce::File& outputFile)
{
    midiGenerator->exportToFile(midiTracks, outputFile);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new VocalMIDIAudioProcessor();
}
