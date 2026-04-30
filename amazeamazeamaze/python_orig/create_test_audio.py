# create_test_audio.py
import numpy as np
import wave
import struct

RATE = 44100  
CHUNK = 8192
DURATION = 0.5  # seconds per "note"

def create_wav(filename, frequencies, pause_between=0.1):
    """Create a WAV file with a sequence of frequencies"""
    
    # Generate audio data
    audio_data = []
    
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
create_wav('sentence1.wav', [4699, 3322, 4699, 349, 1865, 208, 104])

## rocky confuse why humans dumb question
#create_wav('test_sentence.wav', [4699])

## humans go to moon amaze amaze amaze
#create_wav('test_sentence.wav', [261.6, 293.7, 131])


print("\nTest WAV files created!")
