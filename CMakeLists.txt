cmake_minimum_required(VERSION 3.22)

project(VocalMIDIPlugin VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add JUCE as a subdirectory (you'll need to clone JUCE into this folder)
# git clone https://github.com/juce-framework/JUCE.git external/JUCE
add_subdirectory(external/JUCE)

# ONNX Runtime Configuration
set(ONNXRUNTIME_ROOT ${CMAKE_SOURCE_DIR}/external/onnxruntime)
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# Plugin configuration
set(PLUGIN_NAME "VocalMIDI")
set(PLUGIN_MANUFACTURER_CODE "VMFX")
set(PLUGIN_CODE "Vmid")
set(PLUGIN_FORMATS VST3 AU Standalone)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/audio
    ${CMAKE_SOURCE_DIR}/src/ml
    ${CMAKE_SOURCE_DIR}/src/ui
    ${CMAKE_SOURCE_DIR}/src/midi
    ${CMAKE_SOURCE_DIR}/src/performance
    ${CMAKE_SOURCE_DIR}/src/threading
)

# Source files
set(SOURCE_FILES
    src/PluginProcessor.cpp
    src/PluginEditor.cpp
    src/audio/AudioCapture.cpp
    src/audio/FeatureExtraction.cpp
    src/audio/PitchDetector.cpp
    src/audio/RhythmAnalyzer.cpp
    src/ml/ModelInference.cpp
    src/ml/ONNXWrapper.cpp
    src/ml/TensorConverter.cpp
    src/ml/GenreClassifier.cpp
    src/ml/TimbreEncoder.cpp
    src/midi/MIDIGenerator.cpp
    src/midi/MIDITrack.cpp
    src/midi/Quantizer.cpp
    src/ui/PianoRollComponent.cpp
    src/ui/TrackLaneComponent.cpp
    src/ui/ControlPanel.cpp
    src/ui/InstrumentSelector.cpp
    src/performance/PerformanceProfiler.cpp
    src/threading/AudioMLBridge.cpp
)

# Header files
set(HEADER_FILES
    src/PluginProcessor.h
    src/PluginEditor.h
    src/audio/AudioCapture.h
    src/audio/FeatureExtraction.h
    src/audio/PitchDetector.h
    src/audio/RhythmAnalyzer.h
    src/ml/ModelInference.h
    src/ml/ONNXWrapper.h
    src/ml/GenreClassifier.h
    src/ml/TimbreEncoder.h
    src/ml/TensorConverter.h
    src/midi/MIDIGenerator.h
    src/midi/MIDITrack.h
    src/midi/Quantizer.h
    src/ui/PianoRollComponent.h
    src/ui/TrackLaneComponent.h
    src/ui/ControlPanel.h
    src/ui/InstrumentSelector.h
    src/performance/PerformanceProfiler.h
    src/threading/AudioMLBridge.h
)

# Create the plugin target
juce_add_plugin(${PLUGIN_NAME}
    COMPANY_NAME "YourCompany"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT TRUE
    IS_MIDI_EFFECT TRUE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    COPY_PLUGIN_AFTER_BUILD TRUE
    PLUGIN_MANUFACTURER_CODE ${PLUGIN_MANUFACTURER_CODE}
    PLUGIN_CODE ${PLUGIN_CODE}
    FORMATS ${PLUGIN_FORMATS}
    PRODUCT_NAME ${PLUGIN_NAME}
)

# Add source files to target
target_sources(${PLUGIN_NAME}
    PRIVATE
        ${SOURCE_FILES}
        ${HEADER_FILES}
)

# Link JUCE modules and ONNX Runtime
target_link_libraries(${PLUGIN_NAME}
    PRIVATE
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_plugin_client
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_dsp
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
        onnxruntime
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# Compiler definitions
target_compile_definitions(${PLUGIN_NAME}
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0
        JUCE_REPORT_APP_USAGE=0
)

# Copy models to build directory
add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/models $<TARGET_FILE_DIR:${PLUGIN_NAME}>/models
)
