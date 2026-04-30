// ============================================================================
// USAGE: Compile: nvcc CPU_alltime.cu -o CPU_alltime
//        Run WAV: ./CPU_alltime sentence1.wav
// ============================================================================

#include <stdio.h>      // For printf, fopen, fclose, fread, fseek
#include <stdlib.h>     // For malloc, free, exit, atof
#include <string.h>     // For strtok, strcpy, strlen
#include <math.h>       // For cos, sin, sqrt, fabs, M_PI
#include <time.h>       // For time, clock, clock_t (timing)

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
// SECTION 4: FFT 
// ============================================================================

float complex_magnitude_simple(float real, float imag) {
    return sqrt(real * real + imag * imag);
}

void fft_simple(float real_parts[], float imag_parts[], int n) {
    if (n <= 1) {
        return;
    }
    
    int half = n / 2;
    
    float *even_real = (float*)malloc(half * sizeof(float));
    float *even_imag = (float*)malloc(half * sizeof(float));
    float *odd_real = (float*)malloc(half * sizeof(float));
    float *odd_imag = (float*)malloc(half * sizeof(float));
    
    for (int i = 0; i < half; i++) {
        even_real[i] = real_parts[2*i];
        even_imag[i] = imag_parts[2*i];
        odd_real[i] = real_parts[2*i + 1];
        odd_imag[i] = imag_parts[2*i + 1];
    }
    
    fft_simple(even_real, even_imag, half);
    fft_simple(odd_real, odd_imag, half);
    
    for (int k = 0; k < half; k++) {
        float angle = -2.0 * M_PI * k / n;
        float twiddle_real = cos(angle);
        float twiddle_imag = sin(angle);
        
        float t_real = odd_real[k] * twiddle_real - odd_imag[k] * twiddle_imag;
        float t_imag = odd_real[k] * twiddle_imag + odd_imag[k] * twiddle_real;
        
        float saved_even_real = even_real[k];
        float saved_even_imag = even_imag[k];
        
        real_parts[k] = even_real[k] + t_real;
        imag_parts[k] = even_imag[k] + t_imag;
        real_parts[k + half] = saved_even_real - t_real;
        imag_parts[k + half] = saved_even_imag - t_imag;
    }
    
    free(even_real); free(even_imag);
    free(odd_real); free(odd_imag);
}

// ============================================================================
// SECTION 5: AUDIO PROCESSING FUNCTIONS
// ============================================================================

float find_loudest_frequency(float real_parts[], float imag_parts[], int fft_size) {
    int best_index = 0;
    float best_amplitude = 0.0;
    
    for (int i = 0; i < fft_size / 2; i++) {
        float amplitude = complex_magnitude_simple(real_parts[i], imag_parts[i]);
        if (amplitude > best_amplitude) {
            best_amplitude = amplitude;
            best_index = i;
        }
    }
    
    float detected_freq = (float)best_index * RATE / fft_size;
    return detected_freq;
}

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

void calculate_fundamental(short *data, float *out_fft_time, float *out_amp_time, float *out_max_time) {
    // Create arrays for FFT
    float fft_real[FFT_SIZE];
    float fft_imag[FFT_SIZE];
    
    for (int i = 0; i < FFT_SIZE; i++) {
        fft_real[i] = (float)data[i];
        fft_imag[i] = 0.0;
    }
    
    // Time the FFT
    clock_t fft_start = clock();
    fft_simple(fft_real, fft_imag, FFT_SIZE);
    clock_t fft_end = clock();
    *out_fft_time = (float)(fft_end - fft_start) / CLOCKS_PER_SEC;
    
    // Time the amplitude calculation + max finding (find_loudest_frequency)
    clock_t amp_start = clock();
    float current_frequency = find_loudest_frequency(fft_real, fft_imag, FFT_SIZE);
    clock_t amp_end = clock();
    *out_amp_time = (float)(amp_end - amp_start) / CLOCKS_PER_SEC;
    
    // Time the single amplitude check (getting amplitude at detected frequency)
    clock_t max_start = clock();
    int freq_index = (int)(current_frequency * FFT_SIZE / RATE);
    float amplitude = complex_magnitude_simple(fft_real[freq_index], fft_imag[freq_index]);
    clock_t max_end = clock();
    *out_max_time = (float)(max_end - max_start) / CLOCKS_PER_SEC;
    
    // Frequency detection and word matching (not timed)
    if (amplitude > AMPLITUDE_THRESHOLD) {
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
// SECTION 6: WAV FILE READING FUNCTION
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
// SECTION 7: MAIN FUNCTION WITH TIMING
// ============================================================================

int main(int argc, char *argv[]) {
    printf("Chord Translator - C Version\n");
    printf("=====================================\n\n");
    
    if (argc < 2) {
        printf("Usage: %s <wav_file>\n", argv[0]);
        return 1;
    }
    
    load_vocabulary();
    printf("\n");
    
    char *wav_filename = argv[1];
    printf("Processing WAV file: %s\n", wav_filename);
    
    int total_samples = 0, file_rate = 0;
    short *audio_data = read_wav_file(wav_filename, &total_samples, &file_rate);
    
    int total_chunks = total_samples / CHUNK;
    printf("Processing %d chunks...\n", total_chunks);
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
        calculate_fundamental(chunk_data, &chunk_fft_time, &chunk_amp_time, &chunk_max_time);
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
    printf("CPU TIMING BREAKDOWN:\n");
    printf("  FFT Time:           %.6f seconds\n", total_fft_time);
    printf("  Amplitude Time:     %.6f seconds\n", total_amp_time);
    printf("  Max Find Time:      %.6f seconds\n", total_max_time);
    printf("  Total Compute Time: %.6f seconds\n", total_fft_time + total_amp_time + total_max_time);
    printf("Chunks processed: %d\n", total_chunks);
    printf("========================================\n");
    
    free(audio_data);
    return 0;
}
