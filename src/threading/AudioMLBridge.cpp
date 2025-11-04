#include "AudioMLBridge.h"
#include "ml/ModelInference.h"
#include "audio/PitchDetector.h"
#include "audio/RhythmAnalyzer.h"

// ==================== AudioMLBridge ====================

AudioMLBridge::AudioMLBridge()
    : audioQueue(AUDIO_QUEUE_SIZE),
      midiQueue(MIDI_QUEUE_SIZE)
{
}

AudioMLBridge::~AudioMLBridge()
{
}

bool AudioMLBridge::sendAudioData(AudioDataPacket&& packet)
{
    bool success = audioQueue.push(std::move(packet));
    
    if (!success)
    {
        audioOverflows++;
        DBG("Warning: Audio queue overflow!");
    }
    
    return success;
}

bool AudioMLBridge::receiveAudioData(AudioDataPacket& packet)
{
    return audioQueue.pop(packet);
}

bool AudioMLBridge::sendMIDIResult(MIDIResultPacket&& packet)
{
    bool success = midiQueue.push(std::move(packet));
    
    if (!success)
    {
        midiOverflows++;
        DBG("Warning: MIDI queue overflow!");
    }
    
    return success;
}

bool AudioMLBridge::receiveMIDIResult(MIDIResultPacket& packet)
{
    return midiQueue.pop(packet);
}

AudioMLBridge::QueueStats AudioMLBridge::getStats() const
{
    QueueStats stats;
    stats.audioQueueSize = audioQueue.size();
    stats.midiQueueSize = midiQueue.size();
    stats.audioQueueCapacity = audioQueue.getCapacity();
    stats.midiQueueCapacity = midiQueue.getCapacity();
    stats.audioQueueOverflows = audioOverflows.load();
    stats.midiQueueOverflows = midiOverflows.load();
    
    return stats;
}

void AudioMLBridge::resetStats()
{
    audioOverflows = 0;
    midiOverflows = 0;
}

// ==================== MLInferenceThread ====================

MLInferenceThread::MLInferenceThread(AudioMLBridge& bridge, ModelInference& modelInference)
    : Thread("ML Inference"),
      bridge(bridge),
      modelInference(modelInference)
{
}

MLInferenceThread::~MLInferenceThread()
{
    stopThread(2000); // Give it 2 seconds to stop gracefully
}

void MLInferenceThread::run()
{
    DBG("ML Inference thread started");
    
    // Set thread priority lower than audio thread
    #if JUCE_MAC || JUCE_IOS
    // macOS/iOS specific priority setting
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
    #elif JUCE_WINDOWS
    // Windows: Set to below normal priority
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    #elif JUCE_LINUX
    // Linux: Set nice value
    nice(5); // Lower priority
    #endif
    
    AudioDataPacket audioPacket;
    
    while (!threadShouldExit())
    {
        if (!isEnabled.load())
        {
            // Sleep when disabled
            wait(100);
            continue;
        }
        
        // Try to receive audio data from queue
        if (bridge.receiveAudioData(audioPacket))
        {
            // Process audio data through ML models
            auto startTime = juce::Time::getMillisecondCounterHiRes();
            
            GeneratedMIDI result = modelInference.generateMIDI(
                audioPacket.melSpectrogram,
                audioPacket.pitchInfo,
                audioPacket.rhythmInfo,
                currentGenre,
                currentBPM,
                "neutral", // mood
                currentInstrument
            );
            
            auto endTime = juce::Time::getMillisecondCounterHiRes();
            auto inferenceTime = endTime - startTime;
            
            if (inferenceTime > 10.0) // Log if inference takes > 10ms
            {
                DBG("ML inference took " + juce::String(inferenceTime, 2) + " ms");
            }
            
            // Send result back to audio thread
            MIDIResultPacket resultPacket;
            resultPacket.midiData = std::move(result);
            resultPacket.samplePosition = audioPacket.samplePosition;
            resultPacket.timestamp = juce::Time::getMillisecondCounterHiRes();
            
            if (!bridge.sendMIDIResult(std::move(resultPacket)))
            {
                DBG("Warning: Failed to send MIDI result - queue full");
            }
        }
        else
        {
            // No data available, sleep briefly
            wait(1); // 1ms sleep
        }
    }
    
    DBG("ML Inference thread stopped");
}
