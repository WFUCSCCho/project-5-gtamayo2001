# run requirements
pip install -r requirements.txt

# Set environment variables so the compiler can find PortAudio
export CFLAGS="-I$HOME/portaudio/include"
export LDFLAGS="-L$HOME/portaudio/lib"
export LD_LIBRARY_PATH="$HOME/portaudio/lib:$LD_LIBRARY_PATH"

# Now install PyAudio with user permissions (no sudo needed)
pip install --user pyaudio

# Test that it worked
python3 -c "import pyaudio; print('PyAudio installed successfully!')"
