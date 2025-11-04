#include "ModelInference.h"
#include "audio/PitchDetector.h"
#include "audio/RhythmAnalyzer.h"

ModelInference::ModelInference()
    : modelsLoaded(false)
{
    // Model directory will be next to the plugin binary
    auto exePath = juce::File::getSpecialLocation(juce::File::currentExecutableFile);
    modelsDirectory = exePath.getParentDirectory().getChildFile("models");
    
    // Initialize ONNX Runtime environment
    try
    {
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "VocalMIDI");
        sessionOptions = std::make_unique<Ort::SessionOptions>();
        
        // Configure session options for optimal performance
        sessionOptions->SetIntraOpNumThreads(1); // Single thread for low latency
        sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Enable CUDA if available
        #ifdef USE_CUDA
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        sessionOptions->AppendExecutionProvider_CUDA(cuda_options);
        #endif
    }
    catch (const Ort::Exception& e)
    {
        DBG("ONNX Runtime initialization failed: " + juce::String(e.what()));
    }
}

ModelInference::~ModelInference()
{
    // Clean up sessions
    pitchSession.reset();
    contextSession.reset();
    timbreSession.reset();
    drumSession.reset();
    bassSession.reset();
    chordSession.reset();
    melodySession.reset();
    continuationSession.reset();
}

bool ModelInference::loadModels()
{
    try
    {
        if (!modelsDirectory.exists())
        {
            DBG("Models directory not found: " + modelsDirectory.getFullPathName());
            return false;
        }
        
        // Define model files
        struct ModelConfig {
            juce::String name;
            juce::File file;
            std::unique_ptr<Ort::Session>* session;
        };
        
        std::vector<ModelConfig> models = {
            {"pitch_model", modelsDirectory.getChildFile("pitch_model.onnx"), &pitchSession},
            {"context_model", modelsDirectory.getChildFile("context_model.onnx"), &contextSession},
            {"timbre_model", modelsDirectory.getChildFile("timbre_model.onnx"), &timbreSession},
            {"drum_generator", modelsDirectory.getChildFile("drum_generator.onnx"), &drumSession},
            {"bass_generator", modelsDirectory.getChildFile("bass_generator.onnx"), &bassSession},
            {"chord_generator", modelsDirectory.getChildFile("chord_generator.onnx"), &chordSession},
            {"melody_generator", modelsDirectory.getChildFile("melody_generator.onnx"), &melodySession},
            {"continuation_model", modelsDirectory.getChildFile("continuation_model.onnx"), &continuationSession}
        };
        
        DBG("Loading ML models from: " + modelsDirectory.getFullPathName());
        
        int loadedCount = 0;
        for (auto& model : models)
        {
            if (loadModel(model.name, model.file, model.session))
            {
                loadedCount++;
            }
            else
            {
                DBG("Warning: Failed to load " + model.name);
            }
        }
        
        modelsLoaded = (loadedCount >= 3); // Need at least pitch, context, and one generator
        
        if (!modelsLoaded)
        {
            DBG("Warning: Insufficient models loaded. Need at least 3, got " + juce::String(loadedCount));
        }
        else
        {
            DBG("Successfully loaded " + juce::String(loadedCount) + "/" + juce::String(models.size()) + " models");
        }
        
        return modelsLoaded;
    }
    catch (const std::exception& e)
    {
        DBG("Error loading models: " + juce::String(e.what()));
        return false;
    }
}

bool ModelInference::loadModel(const juce::String& modelName, const juce::File& modelFile, std::unique_ptr<Ort::Session>* session)
{
    if (!modelFile.existsAsFile())
    {
        DBG("Model file not found: " + modelFile.getFullPathName());
        return false;
    }
    
    try
    {
        // Convert file path to wide string for ONNX Runtime
        #ifdef _WIN32
        auto modelPath = modelFile.getFullPathName().toWideCharPointer();
        #else
        auto modelPath = modelFile.getFullPathName().toUTF8();
        #endif
        
        // Create ONNX Runtime session
        #ifdef _WIN32
        *session = std::make_unique<Ort::Session>(*env, modelPath, *sessionOptions);
        #else
        *session = std::make_unique<Ort::Session>(*env, modelPath, *sessionOptions);
        #endif
        
        // Log model metadata
        auto inputCount = (*session)->GetInputCount();
        auto outputCount = (*session)->GetOutputCount();
        
        DBG("Loaded model: " + modelName);
        DBG("  Inputs: " + juce::String(inputCount));
        DBG("  Outputs: " + juce::String(outputCount));
        
        // Get input/output names for later use
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < inputCount; ++i)
        {
            auto inputName = (*session)->GetInputNameAllocated(i, allocator);
            DBG("  Input " + juce::String(i) + ": " + juce::String(inputName.get()));
        }
        
        return true;
    }
    catch (const Ort::Exception& e)
    {
        DBG("ONNX Runtime error loading " + modelName + ": " + juce::String(e.what()));
        return false;
    }
}

GeneratedMIDI ModelInference::generateMIDI(
    const std::vector<std::vector<float>>& melSpectrogram,
    const PitchInfo& pitchInfo,
    const RhythmInfo& rhythmInfo,
    const juce::String& genre,
    float bpm,
    const juce::String& mood,
    const juce::String& primaryInstrument)
{
    GeneratedMIDI result;
    result.genre = genre;
    result.instrument = primaryInstrument;
    result.bpm = bpm;
    
    if (!modelsLoaded || melSpectrogram.empty())
    {
        return result;
    }
    
    try
    {
        int numFrames = static_cast<int>(melSpectrogram.size());
        
        // Prepare input tensor from mel-spectrogram
        auto input = prepareInput(melSpectrogram);
        
        // Run pitch model inference
        std::vector<float> pitchOutput;
        if (pitchSession)
        {
            runInference(pitchSession.get(), input, pitchOutput);
            
            // Reshape output to [time][128] pitch probabilities
            if (pitchOutput.size() >= numFrames * 128)
            {
                result.pitch.resize(numFrames, std::vector<float>(128, 0.0f));
                for (int t = 0; t < numFrames; ++t)
                {
                    for (int n = 0; n < 128; ++n)
                    {
                        result.pitch[t][n] = pitchOutput[t * 128 + n];
                    }
                }
            }
        }
        
        // Run context model for velocity and timing
        std::vector<float> contextOutput;
        if (contextSession)
        {
            runInference(contextSession.get(), input, contextOutput);
            
            // Extract velocity predictions
            result.velocity.resize(numFrames, std::vector<float>(128, 0.0f));
            if (contextOutput.size() >= numFrames * 128)
            {
                for (int t = 0; t < numFrames; ++t)
                {
                    for (int n = 0; n < 128; ++n)
                    {
                        result.velocity[t][n] = contextOutput[t * 128 + n] * 127.0f; // Scale to MIDI velocity
                    }
                }
            }
        }
        
        // Use rhythm analyzer output for onset/offset detection
        result.onset.resize(numFrames, 0.0f);
        result.offset.resize(numFrames, 0.0f);
        
        if (!rhythmInfo.onsetEnvelope.empty())
        {
            // Use onset envelope to mark probable onsets
            int envelopeSize = static_cast<int>(rhythmInfo.onsetEnvelope.size());
            for (int t = 0; t < numFrames && t < envelopeSize; ++t)
            {
                result.onset[t] = rhythmInfo.onsetEnvelope[t];
            }
        }
        
        // Fallback: Use detected pitch if models didn't produce valid output
        if (result.pitch.empty() && pitchInfo.midiNote > 0 && pitchInfo.midiNote < 128)
        {
            result.pitch.resize(numFrames, std::vector<float>(128, 0.0f));
            result.velocity.resize(numFrames, std::vector<float>(128, 0.0f));
            
            for (int frame = 0; frame < numFrames; ++frame)
            {
                result.pitch[frame][pitchInfo.midiNote] = pitchInfo.confidence;
                result.velocity[frame][pitchInfo.midiNote] = 80.0f; // Default velocity
            }
        }
        
        return result;
    }
    catch (const std::exception& e)
    {
        DBG("Error in generateMIDI: " + juce::String(e.what()));
        return result;
    }
}

std::vector<GeneratedMIDI> ModelInference::generateMultiTrack(
    const std::vector<std::vector<float>>& melSpectrogram,
    const juce::String& genre,
    float bpm)
{
    std::vector<GeneratedMIDI> tracks;
    
    if (!modelsLoaded)
    {
        return tracks;
    }
    
    // Generate each track type
    std::vector<juce::String> trackTypes = {"Drums", "Bass", "Chords", "Melody"};
    
    for (const auto& trackType : trackTypes)
    {
        GeneratedMIDI track;
        track.genre = genre;
        track.instrument = trackType;
        track.bpm = bpm;
        
        // Run appropriate generator model
        // This would use the GAN models in production
        
        tracks.push_back(track);
    }
    
    return tracks;
}

GeneratedMIDI ModelInference::regenerateTrack(
    const juce::String& trackType,
    const std::vector<std::vector<float>>& contextFeatures)
{
    GeneratedMIDI result;
    result.instrument = trackType;
    
    if (!modelsLoaded)
    {
        return result;
    }
    
    // Use continuation model to regenerate with variation
    auto input = prepareInput(contextFeatures);
    
    // Run inference with appropriate model based on track type
    
    return result;
}

std::vector<float> ModelInference::prepareInput(const std::vector<std::vector<float>>& features)
{
    std::vector<float> flattened;
    
    for (const auto& frame : features)
    {
        flattened.insert(flattened.end(), frame.begin(), frame.end());
    }
    
    return flattened;
}

void ModelInference::runInference(Ort::Session* session, const std::vector<float>& input, std::vector<float>& output)
{
    if (!session)
    {
        DBG("Invalid session for inference");
        return;
    }
    
    try
    {
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Get input/output info
        auto inputName = session->GetInputNameAllocated(0, allocator);
        auto outputName = session->GetOutputNameAllocated(0, allocator);
        
        // Create input tensor
        std::vector<int64_t> inputShape = {1, static_cast<int64_t>(input.size())};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            const_cast<float*>(input.data()),
            input.size(),
            inputShape.data(),
            inputShape.size()
        );
        
        // Run inference
        const char* inputNames[] = {inputName.get()};
        const char* outputNames[] = {outputName.get()};
        
        auto outputTensors = session->Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1
        );
        
        // Extract output
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        size_t outputSize = 1;
        for (auto dim : outputShape)
        {
            outputSize *= dim;
        }
        
        output.assign(outputData, outputData + outputSize);
    }
    catch (const Ort::Exception& e)
    {
        DBG("ONNX Runtime inference error: " + juce::String(e.what()));
    }
}
