// ============================================================================
// USAGE: Compile: nvcc GPU_alltime.cu -o GPU_alltime -lcufft
//        Run WAV: ./GPU_alltime sentence1.wav
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cufft.h>

// ============================================================================
// SECTION 1: CONSTANTS 
// ============================================================================

#define CHUNK 8192
#define RATE 44100
#define AMPLITUDE_THRESHOLD 1000000
#define CHANGING_THRESHOLD 15
#define READING_CLEARANCE 10
#define PHRASE_TIME_LIMIT_SECONDS 2
#define VOCAB_FILE "vocab.csv"
#define MAX_VOCAB_SIZE 200
#define MAX_SENTENCE_LENGTH 50
#define MAX_WORD_LENGTH 50
#define FFT_SIZE CHUNK
#define THREADS_PER_BLOCK 256

// ============================================================================
// SECTION 2: GLOBAL VARIABLES 
// ============================================================================

int can_read_note = 1;
float last_frequency = 0.0;
char current_sentence[MAX_SENTENCE_LENGTH][MAX_WORD_LENGTH];
int sentence_word_count = 0;

time_t last_note_time;
int timer_active = 0;
int hide_freq = 0;

cufftHandle fft_plan;

// ============================================================================
// SECTION 3: VOCABULARY 
// ============================================================================

float vocab_frequencies[MAX_VOCAB_SIZE];
char vocab_words[MAX_VOCAB_SIZE][MAX_WORD_LENGTH];
int vocab_size = 0;

void load_vocabulary() {
    FILE *file = fopen(VOCAB_FILE, "r");
    if (file == NULL) {
        printf("Error: Cannot open %s\n", VOCAB_FILE);
        exit(1);
    }
    
    char line[256];
    
    while (fgets(line, sizeof(line), file) != NULL && vocab_size < MAX_VOCAB_SIZE) {
        char *token = strtok(line, ",");
        if (token == NULL) continue;
        
        float freq = atof(token);
        vocab_frequencies[vocab_size] = freq;
        
        token = strtok(NULL, ",");
        if (token == NULL) continue;
        
        strcpy(vocab_words[vocab_size], token);
        
        int len = strlen(vocab_words[vocab_size]);
        if (len > 0 && vocab_words[vocab_size][len-1] == '\n') {
            vocab_words[vocab_size][len-1] = '\0';
        }
        
        vocab_size++;
    }
    
    fclose(file);
    printf("Loaded %d vocabulary entries\n", vocab_size);
}

// ============================================================================
// SECTION 4: GPU KERNELS 
// ============================================================================

__global__ void convert_short_to_float(short *d_audio, float *d_real, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Bounds check: only process if this threads index is within the array
    if (idx < n) {
        d_real[idx] = (float)d_audio[idx];
    }
}

__global__ void calculate_amplitudes_interleaved(float *d_fft_data, float *d_amplitudes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // process first half of FFT output (positive freqs).
    // second half is a mirror image (negative freqs) and ignored.
    if (idx < n / 2) {
        float real_val = d_fft_data[idx * 2];
        float imag_val = d_fft_data[idx * 2 + 1];
        // pythagorean theorem
        d_amplitudes[idx] = sqrtf(real_val * real_val + imag_val * imag_val);
    }
}

__global__ void find_max_with_shared_memory(float *d_amplitudes, float *d_block_maxes, int *d_block_indices, int n) {
    __shared__ float shared_amps[THREADS_PER_BLOCK];
    __shared__ int shared_indices[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    
    if (global_idx < n / 2) {
        // real data
        shared_amps[tid] = d_amplitudes[global_idx];
        shared_indices[tid] = global_idx;
    } else {
        // sentinel stuff
        shared_amps[tid] = -1.0f; // no real data
        shared_indices[tid] = -1; // invadlid index
    }
    __syncthreads();

    // big help from deepseek
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            // Compare this thread's value with the value "stride" positions away.
            if (shared_amps[tid] < shared_amps[tid + stride]) {
                shared_amps[tid] = shared_amps[tid + stride];
                shared_indices[tid] = shared_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    // After reduction, shared_amps[0] holds the maximum in this block.
    // Only thread 0 writes the result to avoid redundant memory writes.
    if (tid == 0) {
        d_block_maxes[blockIdx.x] = shared_amps[0];
        d_block_indices[blockIdx.x] = shared_indices[0];
    }
}

// ============================================================================
// SECTION 5: AUDIO PROCESSING FUNCTIONS
// ============================================================================

void pick_word(float freq) {
    int best_index = -1;
    float best_distance = READING_CLEARANCE + 999;
    
    for (int i = 0; i < vocab_size; i++) {
        float distance = fabs(freq - vocab_frequencies[i]);
        if (distance <= READING_CLEARANCE && distance < best_distance) {
            best_distance = distance;
            best_index = i;
        }
    }
    
    if (best_index >= 0) {
        if (sentence_word_count < MAX_SENTENCE_LENGTH) {
            strcpy(current_sentence[sentence_word_count], vocab_words[best_index]);
            sentence_word_count++;
            time(&last_note_time);
            timer_active = 1;
        }
    }
}

void sentence_finished() {
    if (sentence_word_count == 0) { timer_active = 0; return; }
    
    for (int i = 0; i < sentence_word_count; i++) {
        printf("%s", current_sentence[i]);
        if (i < sentence_word_count - 1) printf(" ");
    }
    printf("\n");
    
    sentence_word_count = 0;
    timer_active = 0;
}

void check_timer() {
    if (timer_active == 0) return;
    
    time_t current_time;
    time(&current_time);
    float elapsed = current_time - last_note_time;
    
    if (elapsed >= PHRASE_TIME_LIMIT_SECONDS) {
        sentence_finished();
    }
}

// ============================================================================
// SECTION 6: GPU-BASED PROCESSING
// ============================================================================

// Returns 3 timing values via pointers
void calculate_fundamental_gpu(short *chunk_data, float *out_fft_time, float *out_amp_time, float *out_max_time) {
    // allocate GPU mem for audio input
    short *d_audio;
    cudaMalloc(&d_audio, FFT_SIZE * sizeof(short));
    
    int cufft_output_size = (FFT_SIZE / 2 + 1);
    float *d_fft_data;
    cudaMalloc(&d_fft_data, cufft_output_size * 2 * sizeof(float));
    
    // copy audio data to GPU
    cudaMemcpy(d_audio, chunk_data, FFT_SIZE * sizeof(short), cudaMemcpyHostToDevice);
    
    // convert short to float on GPU
    int num_blocks_convert = (FFT_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    float *d_input_real;
    cudaMalloc(&d_input_real, FFT_SIZE * sizeof(float));
    convert_short_to_float<<<num_blocks_convert, THREADS_PER_BLOCK>>>(d_audio, d_input_real, FFT_SIZE);
    cudaDeviceSynchronize();
    cudaFree(d_audio);
    
    // create CUDA events for timing FFT
    cudaEvent_t fft_start, fft_stop;
    cudaEventCreate(&fft_start);
    cudaEventCreate(&fft_stop);
    
    // time FFT 
    cudaEventRecord(fft_start, 0);
    cufftExecR2C(fft_plan, d_input_real, (cufftComplex*)d_fft_data);
    cudaEventRecord(fft_stop, 0);
    cudaEventSynchronize(fft_stop);
    
    // calculate FFT time
    float fft_time_ms = 0.0f;
    cudaEventElapsedTime(&fft_time_ms, fft_start, fft_stop);
    *out_fft_time = fft_time_ms / 1000.0f;
    
    // clean up 
    cudaEventDestroy(fft_start);
    cudaEventDestroy(fft_stop);
    
    // allocate amplitude array on GPU
    float *d_amplitudes;
    cudaMalloc(&d_amplitudes, (FFT_SIZE / 2) * sizeof(float));
    
    int num_blocks = (FFT_SIZE / 2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // create CUDA events for timing amplitude kernel
    cudaEvent_t amp_start, amp_stop;
    cudaEventCreate(&amp_start);
    cudaEventCreate(&amp_stop);
    
    // time the amplitude calculation kernel
    cudaEventRecord(amp_start, 0);
    calculate_amplitudes_interleaved<<<num_blocks, THREADS_PER_BLOCK>>>(d_fft_data, d_amplitudes, FFT_SIZE);
    cudaEventRecord(amp_stop, 0);
    cudaEventSynchronize(amp_stop);
    
    // calculate amplitude time
    float amp_time_ms = 0.0f;
    cudaEventElapsedTime(&amp_time_ms, amp_start, amp_stop);
    *out_amp_time = amp_time_ms / 1000.0f;
    
    // clean up amplitude events
    cudaEventDestroy(amp_start);
    cudaEventDestroy(amp_stop);
    
    cudaFree(d_input_real);
    cudaFree(d_fft_data);
    
    // find max using shared memory 
    float *d_block_maxes;
    int *d_block_indices;
    cudaMalloc(&d_block_maxes, num_blocks * sizeof(float));
    cudaMalloc(&d_block_indices, num_blocks * sizeof(int));
    
    // create CUDA events for timing max-finding kernel
    cudaEvent_t max_start, max_stop;
    cudaEventCreate(&max_start);
    cudaEventCreate(&max_stop);
    
    // time the max-finding kernel 
    cudaEventRecord(max_start, 0);
    find_max_with_shared_memory<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_amplitudes, d_block_maxes, d_block_indices, FFT_SIZE);
    cudaEventRecord(max_stop, 0);
    cudaEventSynchronize(max_stop);
    
    // calculate max-finding time
    float max_time_ms = 0.0f;
    cudaEventElapsedTime(&max_time_ms, max_start, max_stop);
    *out_max_time = max_time_ms / 1000.0f;
    
    // clean up max events
    cudaEventDestroy(max_start);
    cudaEventDestroy(max_stop);
    
    cudaFree(d_amplitudes);
    
    // final reduction on CPU
    float *h_block_maxes = (float*)malloc(num_blocks * sizeof(float));
    int *h_block_indices = (int*)malloc(num_blocks * sizeof(int));
    
    cudaMemcpy(h_block_maxes, d_block_maxes, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_block_indices, d_block_indices, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_block_maxes);
    cudaFree(d_block_indices);
    
    float max_amplitude = 0.0f;
    int max_index = 0;
    
    for (int i = 0; i < num_blocks; i++) {
        if (h_block_maxes[i] > max_amplitude) {
            max_amplitude = h_block_maxes[i];
            max_index = h_block_indices[i];
        }
    }
    
    free(h_block_maxes);
    free(h_block_indices);
    
    float current_frequency = (float)max_index * RATE / FFT_SIZE;
    
    if (max_amplitude > AMPLITUDE_THRESHOLD) {
        if (can_read_note == 0) {
            if (fabs(current_frequency - last_frequency) > CHANGING_THRESHOLD) {
                can_read_note = 1;
            }
        }
        
        if (can_read_note == 1) {
            if (hide_freq == 0) {
                printf("%.1f\n", current_frequency);
            }
            pick_word(current_frequency);
            last_frequency = current_frequency;
            can_read_note = 0;
        }
    } else {
        can_read_note = 1;
    }
}

// ============================================================================
// SECTION 7: WAV FILE READING FUNCTION 
// ============================================================================

short* read_wav_file(const char *filename, int *total_samples, int *file_rate) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) { printf("Error: Cannot open WAV file '%s'\n", filename); exit(1); }
    
    char chunk_id[4]; int chunk_size; char format[4];
    char subchunk1_id[4]; int subchunk1_size;
    short audio_format, num_channels; int sample_rate, byte_rate;
    short block_align, bits_per_sample;
    char subchunk2_id[4]; int subchunk2_size;
    
    fread(chunk_id, 1, 4, file); fread(&chunk_size, 4, 1, file); fread(format, 1, 4, file);
    fread(subchunk1_id, 1, 4, file); fread(&subchunk1_size, 4, 1, file);
    fread(&audio_format, 2, 1, file); fread(&num_channels, 2, 1, file);
    fread(&sample_rate, 4, 1, file); fread(&byte_rate, 4, 1, file);
    fread(&block_align, 2, 1, file); fread(&bits_per_sample, 2, 1, file);
    
    if (subchunk1_size > 16) fseek(file, subchunk1_size - 16, SEEK_CUR);
    
    fread(subchunk2_id, 1, 4, file); fread(&subchunk2_size, 4, 1, file);
    
    int bytes_per_sample = bits_per_sample / 8;
    int num_samples = subchunk2_size / bytes_per_sample;
    if (num_channels > 1) num_samples = num_samples / num_channels;
    
    printf("WAV Info: %d channels, %d-bit, %d Hz\n", num_channels, bits_per_sample, sample_rate);
    printf("Total samples: %d (%.2f seconds)\n", num_samples, (float)num_samples / sample_rate);
    
    short *audio_data = (short*)malloc(num_samples * sizeof(short));
    for (int i = 0; i < num_samples; i++) {
        if (num_channels == 1) fread(&audio_data[i], sizeof(short), 1, file);
        else { short left, right; fread(&left, sizeof(short), 1, file); fread(&right, sizeof(short), 1, file); audio_data[i] = left; }
    }
    
    fclose(file);
    *total_samples = num_samples;
    *file_rate = sample_rate;
    return audio_data;
}

// ============================================================================
// SECTION 8: MAIN FUNCTION 
// ============================================================================

int main(int argc, char *argv[]) {
    printf("Chord Translator - CUDA GPU Version (cuFFT)\n");
    printf("============================================\n");

    if (argc < 2) {
        printf("Usage: %s <wav_file>\n", argv[0]);
        return 1;
    }
    
    if (cufftPlan1d(&fft_plan, FFT_SIZE, CUFFT_R2C, 1) != CUFFT_SUCCESS) {
        printf("Error: Failed to create cuFFT plan\n");
        return 1;
    }
    
    load_vocabulary();
    printf("\n");
    
    char *wav_filename = argv[1];
    printf("Processing WAV file: %s\n", wav_filename);
    
    int total_samples = 0, file_rate = 0;
    short *audio_data = read_wav_file(wav_filename, &total_samples, &file_rate);
    
    int total_chunks = total_samples / CHUNK;
    printf("Processing %d chunks (FFT on GPU with cuFFT)...\n", total_chunks);
    printf("--------------------------------------------------\n");
    
    // Accumulate separate timing for FFT, amplitude, and max
    float total_fft_time = 0.0f;
    float total_amp_time = 0.0f;
    float total_max_time = 0.0f;
    
    for (int chunk_index = 0; chunk_index < total_chunks; chunk_index++) {
        short chunk_data[CHUNK];
        for (int i = 0; i < CHUNK; i++) {
            chunk_data[i] = audio_data[chunk_index * CHUNK + i];
        }
        
        // Get all three timings from calculate_fundamental
        float chunk_fft_time, chunk_amp_time, chunk_max_time;
        calculate_fundamental_gpu(chunk_data, &chunk_fft_time, &chunk_amp_time, &chunk_max_time);
        total_fft_time += chunk_fft_time;
        total_amp_time += chunk_amp_time;
        total_max_time += chunk_max_time;
        
        check_timer();
    }
    
    printf("--------------------------------------------------\n");
    printf("Finished processing WAV file!\n\n");
    
    if (sentence_word_count > 0) sentence_finished();
    
    // Print all three timing results
    printf("\n========================================\n");
    printf("GPU TIMING BREAKDOWN:\n");
    printf("  cuFFT Time:         %.6f seconds\n", total_fft_time);
    printf("  Amplitude Time:     %.6f seconds\n", total_amp_time);
    printf("  Max Find Time:      %.6f seconds\n", total_max_time);
    printf("  Total Kernel Time:  %.6f seconds\n", total_fft_time + total_amp_time + total_max_time);
    printf("Chunks processed: %d\n", total_chunks);
    printf("========================================\n");
    
    cufftDestroy(fft_plan);
    
    free(audio_data);
    return 0;
}
