#!/bin/bash
# ============================================================================
# FILE: collect_timing_all.sh
# PURPOSE: Run CPU and GPU versions on all test files and save results to CSV
# USAGE: chmod +x collect_timing_all.sh
#        ./collect_timing_all.sh
# ============================================================================

# Output CSV file
CSV_FILE="timing_results_all.csv"

# Write CSV header
echo "version,file,num_words,total_samples,sample_time_seconds,num_chunks,time_per_chunk,fft_time,amplitude_time,max_find_time,total_compute_time" > $CSV_FILE

# Loop through test files t1.wav to t10.wav
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    WAV_FILE="t${i}.wav"
    
    echo "========================================="
    echo "Processing CPU: $WAV_FILE"
    echo "========================================="
    
    # Run CPU version and capture ALL output
    CPU_OUTPUT=$(./CPU_alltime "$WAV_FILE" 2>&1)
    
    # Number of words = file number (t1 = 1 word, t2 = 2 words, etc.)
    NUM_WORDS=$i
    
    # Extract total samples (the number before the parenthesis)
    TOTAL_SAMPLES=$(echo "$CPU_OUTPUT" | grep "Total samples:" | awk '{print $3}')
    
    # Extract sample time in seconds (the number inside parentheses)
    SAMPLE_TIME=$(echo "$CPU_OUTPUT" | grep "Total samples:" | grep -oP '\(\K[0-9.]+(?= seconds\))')
    
    # Extract number of chunks
    NUM_CHUNKS=$(echo "$CPU_OUTPUT" | grep "Chunks processed:" | awk '{print $3}')
    
    # Calculate time per chunk from total compute time / chunks
    TOTAL_COMPUTE=$(echo "$CPU_OUTPUT" | grep "Total Compute Time:" | awk '{print $4}')
    TIME_PER_CHUNK=$(echo "scale=6; $TOTAL_COMPUTE / $NUM_CHUNKS" | bc)
    
    # Extract FFT time
    FFT_TIME=$(echo "$CPU_OUTPUT" | grep "FFT Time:" | awk '{print $3}')
    
    # Extract Amplitude time
    AMP_TIME=$(echo "$CPU_OUTPUT" | grep "Amplitude Time:" | awk '{print $3}')
    
    # Extract Max Find time
    MAX_TIME=$(echo "$CPU_OUTPUT" | grep "Max Find Time:" | awk '{print $4}')
    
    # Debug output
    echo "CPU: words=$NUM_WORDS samples=$TOTAL_SAMPLES time=$SAMPLE_TIME chunks=$NUM_CHUNKS"
    echo "     fft=$FFT_TIME amp=$AMP_TIME max=$MAX_TIME total=$TOTAL_COMPUTE per_chunk=$TIME_PER_CHUNK"
    
    # Write CPU results to CSV
    echo "cpu,$WAV_FILE,$NUM_WORDS,$TOTAL_SAMPLES,$SAMPLE_TIME,$NUM_CHUNKS,$TIME_PER_CHUNK,$FFT_TIME,$AMP_TIME,$MAX_TIME,$TOTAL_COMPUTE" >> $CSV_FILE
    
    echo ""
    echo "========================================="
    echo "Processing GPU: $WAV_FILE"
    echo "========================================="
    
    # Run GPU version and capture ALL output
    GPU_OUTPUT=$(./GPU_alltime "$WAV_FILE" 2>&1)
    
    # Number of words = file number
    NUM_WORDS=$i
    
    # Extract total samples
    TOTAL_SAMPLES=$(echo "$GPU_OUTPUT" | grep "Total samples:" | awk '{print $3}')
    
    # Extract sample time in seconds
    SAMPLE_TIME=$(echo "$GPU_OUTPUT" | grep "Total samples:" | grep -oP '\(\K[0-9.]+(?= seconds\))')
    
    # Extract number of chunks
    NUM_CHUNKS=$(echo "$GPU_OUTPUT" | grep "Chunks processed:" | awk '{print $3}')
    
    # Calculate time per chunk from total kernel time / chunks
    TOTAL_KERNEL=$(echo "$GPU_OUTPUT" | grep "Total Kernel Time:" | awk '{print $4}')
    TIME_PER_CHUNK=$(echo "scale=6; $TOTAL_KERNEL / $NUM_CHUNKS" | bc)
    
    # Extract cuFFT time
    FFT_TIME=$(echo "$GPU_OUTPUT" | grep "cuFFT Time:" | awk '{print $3}')
    
    # Extract Amplitude time
    AMP_TIME=$(echo "$GPU_OUTPUT" | grep "Amplitude Time:" | awk '{print $3}')
    
    # Extract Max Find time
    MAX_TIME=$(echo "$GPU_OUTPUT" | grep "Max Find Time:" | awk '{print $4}')
    
    # Debug output
    echo "GPU: words=$NUM_WORDS samples=$TOTAL_SAMPLES time=$SAMPLE_TIME chunks=$NUM_CHUNKS"
    echo "     fft=$FFT_TIME amp=$AMP_TIME max=$MAX_TIME total=$TOTAL_KERNEL per_chunk=$TIME_PER_CHUNK"
    
    # Write GPU results to CSV
    echo "gpu,$WAV_FILE,$NUM_WORDS,$TOTAL_SAMPLES,$SAMPLE_TIME,$NUM_CHUNKS,$TIME_PER_CHUNK,$FFT_TIME,$AMP_TIME,$MAX_TIME,$TOTAL_KERNEL" >> $CSV_FILE
    
    echo ""
done

echo "========================================="
echo "All results saved to $CSV_FILE"
echo "========================================="

# Print the CSV contents
echo ""
echo "CSV Contents:"
echo "-------------"
cat $CSV_FILE
