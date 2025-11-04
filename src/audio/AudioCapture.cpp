#include "AudioCapture.h"

AudioCapture::AudioCapture()
    : writePosition(0), bufferSize(0), currentSampleRate(44100.0)
{
}

AudioCapture::~AudioCapture()
{
}

void AudioCapture::prepare(double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    bufferSize = static_cast<int>(sampleRate * 10); // 10 seconds buffer
    
    capturedBuffer.setSize(2, bufferSize);
    capturedBuffer.clear();
    writePosition = 0;
}

void AudioCapture::reset()
{
    capturedBuffer.clear();
    writePosition = 0;
}

void AudioCapture::process(const juce::AudioBuffer<float>& buffer)
{
    const int numSamples = buffer.getNumSamples();
    const int numChannels = juce::jmin(buffer.getNumChannels(), capturedBuffer.getNumChannels());
    
    for (int channel = 0; channel < numChannels; ++channel)
    {
        for (int i = 0; i < numSamples; ++i)
        {
            capturedBuffer.setSample(channel, writePosition, buffer.getSample(channel, i));
            writePosition = (writePosition + 1) % bufferSize;
        }
    }
}

void AudioCapture::getRecentAudio(juce::AudioBuffer<float>& destination, int numSamples)
{
    const int numChannels = juce::jmin(destination.getNumChannels(), capturedBuffer.getNumChannels());
    
    int readPos = (writePosition - numSamples + bufferSize) % bufferSize;
    
    for (int channel = 0; channel < numChannels; ++channel)
    {
        for (int i = 0; i < numSamples; ++i)
        {
            int pos = (readPos + i) % bufferSize;
            destination.setSample(channel, i, capturedBuffer.getSample(channel, pos));
        }
    }
}
