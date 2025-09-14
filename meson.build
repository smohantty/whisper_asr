project('whisper_test', 'cpp',
  version : '1.0.0',
  default_options : [
    'cpp_std=c++17',
    'buildtype=release',
    'warning_level=2'
  ]
)

# Import filesystem module
fs = import('fs')

# Build whisper.cpp using CMake in unified build directory
whisper_src_dir = meson.current_source_dir() / 'subprojects' / 'whisper.cpp-1.7.6'
whisper_build_dir = meson.current_build_dir() / 'whisper-build'

# Configure cmake build in unified build directory
message('Configuring whisper.cpp with cmake...')
cmake_configure = run_command('cmake', '-B', whisper_build_dir, '-S', whisper_src_dir, '-G', 'Ninja',
  '-DBUILD_SHARED_LIBS=OFF',
  '-DWHISPER_BUILD_TESTS=OFF',
  '-DWHISPER_BUILD_EXAMPLES=OFF',
  '-DWHISPER_BUILD_SERVER=OFF',
  '-DWHISPER_COREML=OFF',
  '-DWHISPER_OPENVINO=OFF',
  '-DGGML_CUDA=OFF',
  '-DGGML_VULKAN=OFF',
  '-DGGML_METAL=OFF',
  '-DWHISPER_SDL2=OFF',
  '-DWHISPER_FFMPEG=OFF',
  '-DCMAKE_BUILD_TYPE=Release',
  check: true, capture: false)

# Build with cmake
message('Building whisper.cpp with cmake...')
cmake_build = run_command('cmake', '--build', whisper_build_dir, '--config', 'Release',
  check: true, capture: false)

# Get compiler for library finding
cc = meson.get_compiler('cpp')

# Find the built libraries using meson's built-in library finder
whisper_lib = cc.find_library('whisper',
  dirs : [whisper_build_dir / 'src'],
  required : true)

ggml_lib = cc.find_library('ggml',
  dirs : [whisper_build_dir / 'ggml' / 'src'],
  required : true)

ggml_base_lib = cc.find_library('ggml-base',
  dirs : [whisper_build_dir / 'ggml' / 'src'],
  required : true)

ggml_cpu_lib = cc.find_library('ggml-cpu',
  dirs : [whisper_build_dir / 'ggml' / 'src'],
  required : true)

# System dependencies
threads_dep = dependency('threads')
openmp_dep = dependency('openmp', required : false)

# Platform-specific dependencies
host_system = host_machine.system()
system_deps = []

if host_system == 'windows'
  # Windows specific libraries if needed
elif host_system == 'darwin'
  # macOS - link Accelerate framework
  system_deps += [declare_dependency(link_args : ['-framework', 'Accelerate'])]
else
  # Linux - link math library
  math_dep = meson.get_compiler('cpp').find_library('m', required : true)
  system_deps += [math_dep]
endif

# Include directories - use relative paths within current source dir
whisper_inc = include_directories('subprojects/whisper.cpp-1.7.6/include', 'subprojects/whisper.cpp-1.7.6/ggml/include')

# Create whisper dependency that includes all necessary libraries
whisper_dep = declare_dependency(
  dependencies : [whisper_lib, ggml_lib, ggml_base_lib, ggml_cpu_lib, threads_dep, openmp_dep] + system_deps,
  include_directories : whisper_inc
)

# Build the test application using the whisper dependency
test_whisper = executable('test_whisper',
  'test_whisper.cpp',
  dependencies : whisper_dep,
  install : true
)

# Build the AudioStreamer library
audiostreamer_lib = static_library('audiostreamer',
  'AudioStreamer.cpp',
  install : false
)

# Build the WhisperBackend library
whisperbackend_lib = static_library('whisperbackend',
  'WhisperBackend.cpp',
  dependencies : [whisper_dep, threads_dep],
  install : false
)

# Build the livestream ASR test application
livestream_asr = executable('livestream_asr',
  'livestream_asr.cpp',
  dependencies : [whisper_dep, threads_dep],
  link_with : [audiostreamer_lib, whisperbackend_lib],
  install : true
)

# Build the WhisperBackend example application
example_whisper_backend = executable('example_whisper_backend',
  'example_whisper_backend.cpp',
  dependencies : [whisper_dep, threads_dep],
  link_with : [audiostreamer_lib, whisperbackend_lib],
  install : true
)

# Build the language switching demo application
language_switching_demo = executable('language_switching_demo',
  'language_switching_demo.cpp',
  dependencies : [whisper_dep, threads_dep],
  link_with : whisperbackend_lib,
  install : true
)

# Build the builder pattern demo application
builder_pattern_demo = executable('builder_pattern_demo',
  'builder_pattern_demo.cpp',
  dependencies : [whisper_dep, threads_dep],
  link_with : whisperbackend_lib,
  install : true
)

# Copy sample files to build directory if they exist
# Note: Using resources directory since subproject files are sandboxed
resources_dir = meson.current_source_dir() / 'resources'
sample_model = resources_dir / 'ggml-base.en.bin'
sample_audio = resources_dir / 'jfk.wav'

if fs.exists(sample_model)
  configure_file(
    input : sample_model,
    output : 'ggml-base.en.bin',
    copy : true
  )
  message('Copying model file: ggml-base.en.bin')
endif

if fs.exists(sample_audio)
  configure_file(
    input : sample_audio,
    output : 'jfk.wav',
    copy : true
  )
  message('Copying audio file: jfk.wav')
endif

# Print build summary
message('')
message('=== Whisper Test Applications Build ===')
message('Targets: test_whisper, livestream_asr, example_whisper_backend, language_switching_demo, builder_pattern_demo')
message('Language: C++17')
message('Build type: ' + get_option('buildtype'))
message('Install: ' + get_option('prefix'))
message('')
message('Usage after build:')
message('  cd builddir')
message('  ./test_whisper [model_path] [audio_path]')
message('  ./livestream_asr [model_path]')
message('  ./example_whisper_backend [base_model_path]')
message('  ./language_switching_demo [base_model_path]')
message('  ./builder_pattern_demo')
message('')
message('Note: livestream applications require:')
message('  - Working microphone')
message('  - arecord utility (ALSA tools)')
message('  - Model file (e.g., ggml-small.en.bin)')
message('')
message('WhisperBackend API Features:')
message('  - Live streaming ASR with callback interface')
message('  - Multi-language support (English/Korean)')
message('  - Runtime language switching')
message('  - Builder pattern for flexible model configuration')
message('  - Internal audio queue and worker thread')
message('  - Speech tag handling (Start/Continue/End)')
message('  - Thread-safe audio processing')
message('')
