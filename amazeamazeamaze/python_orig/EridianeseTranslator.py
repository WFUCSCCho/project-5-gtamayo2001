import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import csv
from threading import Timer
import anthropic
import argparse
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
# NEW: Added for WAV file support
import wave
import sys
import os

# constants
CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
INPUT_DEVICE_INDEX = 3
AMPLITUDE_THRESHOLD = 1000000
CHANGING_THRESHOLD = 15 # amount that the freq has to change to read new value
READING_CLEARANCE = 2 # amount that freq can be +- the table freq value
PHRASE_TIME_LIMIT_SECONDS = 2 # amount of time to wait after a note before loading into voice
ELEVENLABS_VOICEID = 'iP95p4xoKVk53GoZ742B'

# global variables
can_read_note = True
last_frequency = 0
current_sentence = list()

# argument for disabling ttp
parser = argparse.ArgumentParser()
parser.add_argument('--no-ttp', action='store_true')
parser.add_argument('--no-claude', action='store_true')
parser.add_argument('--hide-spectrum', action='store_true')
parser.add_argument('--hide-freq', action='store_true')
# NEW: Added argument for WAV file input
parser.add_argument('--wav-file', type=str, default=None, 
                    help='Process a pre-recorded WAV file instead of live microphone')
args = parser.parse_args()

# claude
anthropic_client = anthropic.Anthropic() if not args.no_claude and not args.no_ttp else None

# elevenlabs
elevenlabs_client = ElevenLabs() if not args.no_ttp else None

# NEW: Only initialize PyAudio if we're NOT using a WAV file
# When using a WAV file, we don't need microphone access at all
if args.wav_file is None:
    # pyaudio
    p = pyaudio.PyAudio()

    audio_input_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,            
        input=True,
        frames_per_buffer=CHUNK, 
        input_device_index=INPUT_DEVICE_INDEX)
else:
    # NEW: When using WAV file, set these to None so we don't try to access microphone
    p = None
    audio_input_stream = None
    print(f"Processing WAV file: {args.wav_file}")

def plot_spectrum(data):
    fft_data = np.fft.fft(data) # Perform FFT
    abs_fft = np.abs(fft_data) # Absolute Values
    freqs = np.fft.fftfreq(len(data), d = 1 / RATE) # calculate freqs
    plt.clf() 
    plt.plot(freqs, abs_fft)
    plt.xlabel("freqnecy (Hz)")
    plt.ylabel("magnitude")
    plt.title("real-time FFT spectrum")
    plt.xlim(0, RATE/5)
    plt.ylim(0, np.max(abs_fft))
    plt.draw()
    plt.pause(0.001)

def calculate_fundemental(data):
    global can_read_note, last_frequency
    dataLength = len(data)
    fft_data = np.fft.fft(data) # Perform FFT
    abs_fft = np.abs(fft_data) # Absolute Values
    freqs = np.fft.fftfreq(dataLength, d = 1 / RATE) # calculate freqs
    
    # split array because second half is weird negatives
    amplitudes, dud_amplitudes = np.split(abs_fft, 2) 
    frequencies, neg_freqs = np.split(freqs, 2)

    loudestFreqIndex = np.argmax(amplitudes) # find index of loudest amplitude
    if(amplitudes[loudestFreqIndex] > AMPLITUDE_THRESHOLD):
        current_frequency = frequencies[loudestFreqIndex]
        if(not can_read_note):
            if(abs(current_frequency - last_frequency) > CHANGING_THRESHOLD): # only read note change if the change is bigger than threshold
                can_read_note = True
        if(can_read_note):
            if(not args.hide_freq):
                print(current_frequency)
            pick_word(current_frequency)
            last_frequency = current_frequency
            can_read_note = False
    else: # once it goes quiet again, let it reset
        can_read_note = True

def pick_word(freq):
    freq_reference, word = np.genfromtxt('vocab.csv', delimiter=',', dtype=None, unpack=True, encoding='utf-8', usecols=(0, 1))
    for i in range(len(freq_reference)):
        high_end_freq = freq + READING_CLEARANCE
        low_end_freq = freq - READING_CLEARANCE
        if(low_end_freq <= freq_reference[i] <= high_end_freq):
            load_into_sentence(word[i])

def sentence_finished():
    sentence = current_sentence.copy()
    current_sentence.clear()
    if(len(sentence) == 0):
        return
    if(args.no_ttp):
        print(" ".join(sentence))
        return
    
    if(args.no_claude):
        tts_text = " ".join(sentence)
    else:
        message = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            temperature=0,
            system="You are a grammar corrector. Output ONLY the corrected sentence with no explanation or commentary. Example: input 'I Hunger' → output 'I am hungry.'",
            messages=[
                {
                "role": "user",
                "content": " ".join(sentence)
                }
            ]
        )
        tts_text = message.content[0].text
    print(tts_text)
    ttp_stream = elevenlabs_client.text_to_speech.stream(
        text=tts_text,
        voice_id=ELEVENLABS_VOICEID,
        model_id="eleven_turbo_v2_5"
    )
    stream(ttp_stream)

timer = Timer(PHRASE_TIME_LIMIT_SECONDS, sentence_finished)

def load_into_sentence(word):
    global timer
    if(timer.is_alive()):
        timer.cancel()
    timer = Timer(PHRASE_TIME_LIMIT_SECONDS, sentence_finished)
    timer.start()
    current_sentence.append(word)

# NEW: Function to process a WAV file instead of live microphone
def process_wav_file(wav_filename):
    """
    Read a pre-recorded WAV file and process it chunk by chunk,
    exactly like the live microphone version would.
    """
    print(f"Opening WAV file: {wav_filename}")
    
    # NEW: Check if file exists
    if not os.path.exists(wav_filename):
        print(f"Error: File '{wav_filename}' not found!")
        sys.exit(1)
    
    # NEW: Open the WAV file for reading
    try:
        wav_file = wave.open(wav_filename, 'rb')
    except Exception as e:
        print(f"Error opening WAV file: {e}")
        sys.exit(1)
    
    # NEW: Verify WAV file parameters match our expected format
    wav_channels = wav_file.getnchannels()
    wav_sample_width = wav_file.getsampwidth()
    wav_framerate = wav_file.getframerate()
    wav_total_frames = wav_file.getnframes()
    
    print(f"WAV Info: {wav_channels} channels, {wav_sample_width*8}-bit, {wav_framerate} Hz")
    print(f"Total frames: {wav_total_frames} ({wav_total_frames/wav_framerate:.2f} seconds)")
    
    # NEW: Read all audio data from the WAV file
    raw_audio_data = wav_file.readframes(wav_total_frames)
    wav_file.close()
    
    # NEW: Convert raw bytes to numpy array (same format as microphone input)
    audio_data = np.frombuffer(raw_audio_data, dtype=np.int16)
    
    # NEW: If WAV is stereo, convert to mono by taking only first channel
    if wav_channels > 1:
        print("Converting stereo to mono...")
        audio_data = audio_data[::wav_channels]  # Take every other sample (first channel)
    
    # NEW: Process the audio data in chunks, just like the live version
    total_chunks = len(audio_data) // CHUNK
    print(f"Processing {total_chunks} chunks...")
    print("-" * 50)
    
    for chunk_index in range(total_chunks):
        # NEW: Extract one chunk of audio data
        start_idx = chunk_index * CHUNK
        end_idx = start_idx + CHUNK
        chunk_data = audio_data[start_idx:end_idx]
        
        # NEW: Optional - show spectrum for this chunk
        if not args.hide_spectrum:
            plot_spectrum(chunk_data)
        
        # NEW: Process this chunk through the same frequency detection
        # This is EXACTLY what the live microphone does per chunk
        calculate_fundemental(chunk_data)
        
        # NEW: Show progress every 100 chunks
        if chunk_index % 100 == 0 and chunk_index > 0:
            print(f"  Processed {chunk_index}/{total_chunks} chunks...")
    
    print("-" * 50)
    print("Finished processing WAV file!")

# NEW: Main execution - choose between WAV file and live microphone
if args.wav_file is not None:
    # NEW: Process pre-recorded WAV file (works on any machine, no microphone needed)
    process_wav_file(args.wav_file)
    # NEW: Force any pending sentence to be displayed
    if len(current_sentence) > 0:
        sentence_finished()
else:
    # Original live microphone processing
    try:
        while True:
            data = np.frombuffer(audio_input_stream.read(CHUNK), dtype=np.int16)
            if(not args.hide_spectrum):
                plot_spectrum(data)
            calculate_fundemental(data)
    except KeyboardInterrupt:
        pass
    finally:
        audio_input_stream.stop_stream()
        audio_input_stream.close()
        p.terminate()
