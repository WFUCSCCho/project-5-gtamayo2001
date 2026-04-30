# create wav files
#python3 create_test_audio.py

# run with WAV file
module load nvidia/cuda12/cuda/12.8.1
nvcc CPU_alltime.cu -o CPU_alltime
./CPU_alltime sentence1.wav

# run with mic input
#./EridianeseTranslator_CPU 