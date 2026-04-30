# create_test_audio.py
import numpy as np
import wave
import struct

RATE = 44100  
CHUNK = 8192
DURATION = 0.7  # seconds per "note"

def create_wav(filename, frequencies, pause_between=0.5):
    """Create a WAV file with a sequence of frequencies"""
    
    # Generate audio data
    audio_data = []

    # Start with 0.2 seconds of silence
    start_silence = int(RATE * 0.2)
    for i in range(start_silence):
        audio_data.append(0)
    
    for freq in frequencies:
        # Generate tone
        num_samples = int(RATE * DURATION)
        for i in range(num_samples):
            sample = int(16000 * np.sin(2 * np.pi * freq * i / RATE))
            audio_data.append(sample)
        
        # Add silence between notes
        silence_samples = int(RATE * pause_between)
        for i in range(silence_samples):
            audio_data.append(0)
    
    # NEW: Add 1.5 seconds of silence at the end
    # This gives the detector time to finish processing the last word
    # and allows the phrase timer to expire
    end_silence = int(RATE * 1.5)
    for i in range(end_silence):
        audio_data.append(0)
    
    # Write WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(RATE)
        
        # Convert to binary
        for sample in audio_data:
            wav_file.writeframes(struct.pack('<h', sample))
    
    print(f"Created {filename}: {len(audio_data)} samples, {len(audio_data)/RATE:.1f} seconds")

# Create test files
# Simple single word test
#create_wav('test_hello.wav', [440.0])

# Phrase test
#create_wav('test_phrase.wav', [440.0, 554.4, 659.3, 880.0])

# Known vocab words test (adjust frequencies to match your vocab.csv!)
#create_wav('test_sentence.wav', [261.6, 293.7, 329.6])  # C, D, E notes

# sentences
## rocky happy rocky see big human friends
#create_wav('sentence1.wav', [4699, 3322, 4699, 349, 1865, 208, 104])

## rocky confuse why humans dumb question
#create_wav('sentence2.wav', [4699, 1397, 1319, 208, 1760, 5274])

## amaze amaze amaze
#create_wav('t1.wav', [131])
#create_wav('t2.wav', [131,1480])
#create_wav('t3.wav', [131,1480,131])
#create_wav('t4.wav', [131,1480,131,1480])
#create_wav('t5.wav', [131,1480,131,1480,131])
#create_wav('t6.wav', [131,1480,131,1480,131,1480])
#create_wav('t7.wav', [131,1480,131,1480,131,1480,131])
#create_wav('t8.wav', [131,1480,131,1480,131,1480,131,1480])
#create_wav('t9.wav', [131,1480,131,1480,131,1480,131,1480,131])
#create_wav('t10.wav', [131,1480,131,1480,131,1480,131,1480,131,1480])
create_wav('t11.wav', [131,1480,131,1480,131,1480,131,1480,131,1480,131])
create_wav('t12.wav', [131,1480,131,1480,131,1480,131,1480,131,1480,131,1480])
create_wav('t13.wav', [131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131])
create_wav('t14.wav', [131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131,1480])
create_wav('t15.wav', [131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131])
create_wav('t16.wav', [131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131,1480])
create_wav('t17.wav', [131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131])
create_wav('t18.wav', [131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131,1480])
create_wav('t19.wav', [131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131])
create_wav('t20.wav', [131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131,1480,131,1480])



print("\nTest WAV files created!")