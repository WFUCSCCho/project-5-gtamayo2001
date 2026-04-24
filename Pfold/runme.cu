/**
 * runme.cu
 * 
 * SIMPLE VERSION: No structs, just basic arrays.
 * Read starting structure and native contacts.
 * Find the right temperature for 50/50 folding/unfolding.
 * 
 * Run:     module load nvidia/cuda12/cuda/12.8.1
 *          nvcc runme.cu -o runme
 *          ./runme 1r69_T_99.pdb contacts.dat
 */

#include <stdio.h>      // For printf, fopen, fscanf
#include <stdlib.h>     // For rand, srand, abs, atof
#include <math.h>       // For sqrtf, logf, cosf
#include <string.h>     // For strncmp, strcmp
#include <time.h>       // For time()

// ============================================================
// CONSTANTS - Hardcoded numbers
// ============================================================

#define MAX_BEADS 200           // Maximum number of beads (one per residue)
#define MAX_CONTACTS 500        // Maximum number of native contacts

// Q thresholds - fraction of contacts formed (0.0 to 1.0)
#define Q_UNFOLDED 0.2f         // Below this = unfolded
#define Q_FOLDED   0.8f         // Above this = folded

// Simulation parameters
#define DT 0.002f               // Time step
#define MASS 1.0f               // Mass of each bead
#define FRICTION 1.0f           // Friction
#define MAX_STEPS 100000        // Max steps before giving up
#define CONTACT_CUTOFF 1.2f     // Multiplier for contact distance

// Temperature tuning
#define TEST_TRAJECTORIES 10    // Trajectories per temperature
#define TEMP_START 0.01f         // First temperature to try
#define TEMP_END   0.5f         // Last temperature to try
#define TEMP_STEP  0.05f         // Step size

// ============================================================
// GLOBAL ARRAYS - No structs, just simple arrays
// ============================================================

// Bead positions: pos_x[i] is x-coordinate of bead i
float pos_x[MAX_BEADS];
float pos_y[MAX_BEADS];
float pos_z[MAX_BEADS];

// Bead velocities: vel_x[i] is x-velocity of bead i
float vel_x[MAX_BEADS];
float vel_y[MAX_BEADS];
float vel_z[MAX_BEADS];

// Forces on each bead
float force_x[MAX_BEADS];
float force_y[MAX_BEADS];
float force_z[MAX_BEADS];

// Native contacts: each contact is a pair of beads
int contact_bead_i[MAX_CONTACTS];       // First bead in contact pair (0-indexed)
int contact_bead_j[MAX_CONTACTS];       // Second bead in contact pair (0-indexed)
float contact_native_dist[MAX_CONTACTS]; // Ideal distance

// Transition state (saved copy of starting positions)
float ts_x[MAX_BEADS];
float ts_y[MAX_BEADS];
float ts_z[MAX_BEADS];

int num_beads = 0;              // How many beads we have
int num_contacts = 0;           // How many contacts we have

// ============================================================
// BASIC MATH FUNCTIONS
// ============================================================

// Distance between two beads (using their indices)
float distance(int i, int j, float* x, float* y, float* z) {
    float dx = x[i] - x[j];
    float dy = y[i] - y[j];
    float dz = z[i] - z[j];
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

// Random number from Gaussian distribution (mean=0, std=1)
float random_gaussian() {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    
    if (u1 < 0.000001f) u1 = 0.000001f;  // Avoid log(0)
    
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
}

// ============================================================
// FILE READING FUNCTIONS
// ============================================================

// Read transition state from PDB file - ONLY CA ATOMS (one per residue)
int read_transition_pdb(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("ERROR: Cannot open %s\n", filename);
        return 0;
    }
    
    char line[256];
    num_beads = 0;
    
    printf("Reading CA atoms from PDB file...\n");
    
    while (fgets(line, sizeof(line), file)) {
        // Only read ATOM lines
        if (strncmp(line, "ATOM", 4) == 0 || strncmp(line, "HETATM", 6) == 0) {
            
            // Check if this is a CA atom (columns 13-16 in PDB format)
            // PDB format: columns 1-4=ATOM, 13-16=atom name
            char atom_name[5];
            strncpy(atom_name, line + 12, 4);
            atom_name[4] = '\0';
            
            // Remove leading spaces
            char clean_name[5] = "";
            int idx = 0;
            for (int i = 0; i < 4; i++) {
                if (atom_name[i] != ' ') {
                    clean_name[idx++] = atom_name[i];
                }
            }
            clean_name[idx] = '\0';
            
            // Only process CA atoms (alpha carbons, one per residue)
            if (strcmp(clean_name, "CA") == 0) {
                
                // Extract coordinates starting at column 30
                float x, y, z;
                if (sscanf(line + 30, "%f %f %f", &x, &y, &z) == 3) {
                    pos_x[num_beads] = x;
                    pos_y[num_beads] = y;
                    pos_z[num_beads] = z;
                    num_beads++;
                    
                    // Print first few for verification
                    if (num_beads <= 5) {
                        printf("  Bead %d (CA): %.3f %.3f %.3f\n", 
                               num_beads, x, y, z);
                    }
                }
            }
            
            if (num_beads >= MAX_BEADS) break;
        }
    }
    
    fclose(file);
    
    if (num_beads == 0) {
        printf("ERROR: No CA atoms found in %s\n", filename);
        return 0;
    }
    
    printf("Read %d CA beads from %s\n", num_beads, filename);
    return 1;
}

// Read native contacts file
// Format: number_of_contacts
//         bead_i bead_j native_distance
int read_contacts(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("ERROR: Cannot open %s\n", filename);
        return 0;
    }
    
    // First line: total contacts
    if (fscanf(file, "%d", &num_contacts) != 1) {
        printf("ERROR: Could not read contact count from %s\n", filename);
        fclose(file);
        return 0;
    }
    
    printf("File reports %d total contacts\n", num_contacts);
    
    if (num_contacts > MAX_CONTACTS) {
        printf("ERROR: Too many contacts (%d > %d)\n", num_contacts, MAX_CONTACTS);
        fclose(file);
        return 0;
    }
    
    // Read each contact
    for (int i = 0; i < num_contacts; i++) {
        int bi, bj;
        float dist;
        if (fscanf(file, "%d %d %f", &bi, &bj, &dist) != 3) {
            printf("ERROR: Could not read contact %d\n", i+1);
            fclose(file);
            return 0;
        }
        
        // Convert from 1-based (file) to 0-based (arrays)
        contact_bead_i[i] = bi - 1;
        contact_bead_j[i] = bj - 1;
        contact_native_dist[i] = dist;
        
        // Print first few for verification
        if (i < 5) {
            printf("  Contact %d: bead %d -- bead %d : %.2f A\n", 
                   i+1, bi, bj, dist);
        }
    }
    
    fclose(file);
    printf("Successfully read %d native contacts from %s\n", num_contacts, filename);
    return 1;
}

// ============================================================
// SIMULATION FUNCTIONS
// ============================================================

// Calculate Q (absolute number of native contacts formed)
int calculate_q_absolute(float* x, float* y, float* z) {
    int contacts_formed = 0;
    
    // Check each native contact pair
    for (int c = 0; c < num_contacts; c++) {
        int i = contact_bead_i[c];
        int j = contact_bead_j[c];
        float native = contact_native_dist[c];
        
        // Bounds check
        if (i >= num_beads || j >= num_beads) {
            continue;
        }
        
        // Current distance between these two beads
        float dx = x[i] - x[j];
        float dy = y[i] - y[j];
        float dz = z[i] - z[j];
        float current = sqrtf(dx*dx + dy*dy + dz*dz);
        
        // Contact is formed if within cutoff * native distance
        if (current < CONTACT_CUTOFF * native) {
            contacts_formed++;
        }
    }
    
    return contacts_formed;
}

// Calculate Q as a fraction (0.0 to 1.0)
float calculate_q_fraction(float* x, float* y, float* z) {
    int formed = calculate_q_absolute(x, y, z);
    return (float)formed / (float)num_contacts;
}

// Assign random velocities based on temperature
void assign_random_velocities(float* vx, float* vy, float* vz, float temp) {
    float scale = sqrtf(temp / MASS);  // From physics: equipartition theorem
    
    for (int i = 0; i < num_beads; i++) {
        vx[i] = random_gaussian() * scale;
        vy[i] = random_gaussian() * scale;
        vz[i] = random_gaussian() * scale;
    }
}

// Copy transition state to working arrays
void reset_to_transition_state(float* dest_x, float* dest_y, float* dest_z) {
    for (int i = 0; i < num_beads; i++) {
        dest_x[i] = ts_x[i];
        dest_y[i] = ts_y[i];
        dest_z[i] = ts_z[i];
    }
}

// Calculate forces (simple Go-model: only native contacts attract)
void calculate_forces(float* x, float* y, float* z, 
                      float* fx, float* fy, float* fz) {
    // Set all forces to zero first
    for (int i = 0; i < num_beads; i++) {
        fx[i] = 0.0f;
        fy[i] = 0.0f;
        fz[i] = 0.0f;
    }
    
    float spring = 500.0f;  // How strong the attraction is
    
    // Loop over all native contacts
    for (int c = 0; c < num_contacts; c++) {
        int i = contact_bead_i[c];
        int j = contact_bead_j[c];
        float r0 = contact_native_dist[c];
        
        // Bounds check
        if (i >= num_beads || j >= num_beads) {
            continue;
        }
        
        // Vector from i to j
        float dx = x[j] - x[i];
        float dy = y[j] - y[i];
        float dz = z[j] - z[i];
        
        // Current distance
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        if (r < 0.001f) r = 0.001f;  // Avoid division by zero
        
        // Force magnitude: F = -k * (r - r0)
        // Positive = pushing apart, Negative = pulling together
        float force_mag = -spring * (r - r0);
        
        // Direction (unit vector from i to j)
        float ux = dx / r;
        float uy = dy / r;
        float uz = dz / r;
        
        // Add force to bead i (pulls toward j if r > r0)
        fx[i] += force_mag * ux;
        fy[i] += force_mag * uy;
        fz[i] += force_mag * uz;
        
        // Equal and opposite force on bead j (Newton's 3rd law)
        fx[j] -= force_mag * ux;
        fy[j] -= force_mag * uy;
        fz[j] -= force_mag * uz;
    }
}

// Take one simulation step (Langevin dynamics)
void take_step(float* x, float* y, float* z,
               float* vx, float* vy, float* vz,
               float* fx, float* fy, float* fz,
               float temp) {
    
    // Calculate forces based on current positions
    calculate_forces(x, y, z, fx, fy, fz);
    
    // Random force strength (from fluctuation-dissipation theorem)
    float rand_scale = sqrtf(2.0f * FRICTION * temp / DT);
    
    // Update each bead
    for (int i = 0; i < num_beads; i++) {
        // Random forces (Gaussian noise)
        float rand_x = random_gaussian() * rand_scale;
        float rand_y = random_gaussian() * rand_scale;
        float rand_z = random_gaussian() * rand_scale;
        
        // Total force = deterministic + friction + random
        float total_fx = fx[i] - FRICTION * vx[i] + rand_x;
        float total_fy = fy[i] - FRICTION * vy[i] + rand_y;
        float total_fz = fz[i] - FRICTION * vz[i] + rand_z;
        
        // Update velocity (half step)
        vx[i] += 0.5f * DT * total_fx / MASS;
        vy[i] += 0.5f * DT * total_fy / MASS;
        vz[i] += 0.5f * DT * total_fz / MASS;
        
        // Update position (full step)
        x[i] += DT * vx[i];
        y[i] += DT * vy[i];
        z[i] += DT * vz[i];
    }
}

// Run one trajectory, return 1 if folded, 0 if unfolded
int run_single_trajectory(float temp) {
    // Working arrays for this trajectory
    float x[MAX_BEADS], y[MAX_BEADS], z[MAX_BEADS];
    float vx[MAX_BEADS], vy[MAX_BEADS], vz[MAX_BEADS];
    float fx[MAX_BEADS], fy[MAX_BEADS], fz[MAX_BEADS];
    
    // Start from transition state
    reset_to_transition_state(x, y, z);
    
    // Random initial velocities
    assign_random_velocities(vx, vy, vz, temp);
    
    // Calculate initial Q
    int q_start = calculate_q_absolute(x, y, z);
    float q_frac_start = (float)q_start / (float)num_contacts;
    
    // Run simulation
    for (int step = 0; step < MAX_STEPS; step++) {
        // Take a step
        take_step(x, y, z, vx, vy, vz, fx, fy, fz, temp);
        
        // Check Q every 100 steps
        if (step % 100 == 0) {
            int q_abs = calculate_q_absolute(x, y, z);
            float q_frac = (float)q_abs / (float)num_contacts;
            
            if (q_frac < Q_UNFOLDED) {
                return 0;   // Unfolded
            }
            if (q_frac > Q_FOLDED) {
                return 1;   // Folded
            }
        }
    }
    
    // Max steps reached
    printf("Warning: Trajectory hit max steps\n");
    return 0;
}

// ============================================================
// TEMPERATURE TUNING
// ============================================================

// Find temperature that gives ~50/50 split
float find_pfold_temperature() {
    float best_temp = 1.0f;
    int best_diff = 999;
    
    printf("\n--- Temperature Tuning ---\n");
    printf("Temp\tFolded\tUnfolded\tDiff from 5/5\n");
    printf("-----------------------------------------\n");
    
    for (float temp = TEMP_START; temp <= TEMP_END; temp += TEMP_STEP) {
        int folded = 0;
        int unfolded = 0;
        
        for (int t = 0; t < TEST_TRAJECTORIES; t++) {
            int result = run_single_trajectory(temp);
            if (result == 1) folded++;
            else unfolded++;
        }
        
        int expected = TEST_TRAJECTORIES / 2;
        int diff = abs(folded - expected);
        
        printf("%.2f\t%d\t%d\t\t%d\n", temp, folded, unfolded, diff);
        
        if (diff < best_diff) {
            best_diff = diff;
            best_temp = temp;
        }
        
        // Early exit if we found perfect 50/50
        if (diff == 0) {
            printf("Found perfect 50/50 at T=%.2f\n", temp);
            break;
        }
    }
    
    printf("\nBest temperature: %.2f (diff=%d)\n", best_temp, best_diff);
    return best_temp;
}

// ============================================================
// DIAGNOSTIC FUNCTION
// ============================================================

void print_diagnostics() {
    printf("\n--- DIAGNOSTIC INFO ---\n");
    
    // Calculate starting Q
    int q_start_abs = calculate_q_absolute(pos_x, pos_y, pos_z);
    float q_start_frac = (float)q_start_abs / (float)num_contacts;
    
    printf("Number of beads: %d\n", num_beads);
    printf("Number of contacts: %d\n", num_contacts);
    printf("Starting Q (absolute): %d contacts formed\n", q_start_abs);
    printf("Starting Q (fraction): %.3f\n", q_start_frac);
    printf("Unfolded threshold: %.3f (%d contacts)\n", 
           Q_UNFOLDED, (int)(Q_UNFOLDED * num_contacts));
    printf("Folded threshold: %.3f (%d contacts)\n", 
           Q_FOLDED, (int)(Q_FOLDED * num_contacts));
    
    // Check bead index bounds
    int max_bead_in_contacts = 0;
    for (int c = 0; c < num_contacts; c++) {
        if (contact_bead_i[c] > max_bead_in_contacts) 
            max_bead_in_contacts = contact_bead_i[c];
        if (contact_bead_j[c] > max_bead_in_contacts) 
            max_bead_in_contacts = contact_bead_j[c];
    }
    printf("Max bead index in contacts: %d\n", max_bead_in_contacts + 1);
    
    if (max_bead_in_contacts >= num_beads) {
        printf("WARNING: Contact indices exceed number of beads!\n");
        printf("  This means PDB and contacts files don't match.\n");
    }
    
    printf("------------------------\n");
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <transition.pdb> <contacts.dat>\n", argv[0]);
        printf("Example: %s 1r69_T_99.pdb contacts.dat\n", argv[0]);
        return 1;
    }
    
    srand(time(NULL));
    
    printf("========================================\n");
    printf("CHUNK 1.1: Setup (No Structs Version)\n");
    printf("========================================\n\n");
    
    // Step 1: Read transition state
    printf("STEP 1: Reading transition state...\n");
    if (!read_transition_pdb(argv[1])) {
        printf("Failed to read PDB file.\n");
        return 1;
    }
    
    // Step 2: Read contacts
    printf("\nSTEP 2: Reading contacts...\n");
    if (!read_contacts(argv[2])) {
        printf("Failed to read contacts file.\n");
        return 1;
    }
    
    // Step 3: Save transition state copy
    printf("\nSTEP 3: Saving transition state...\n");
    for (int i = 0; i < num_beads; i++) {
        ts_x[i] = pos_x[i];
        ts_y[i] = pos_y[i];
        ts_z[i] = pos_z[i];
    }
    
    // Step 4: Summary
    printf("\nSTEP 4: Summary...\n");
    printf("  Beads: %d\n", num_beads);
    printf("  Contacts: %d\n", num_contacts);
    
    // Print diagnostics
    print_diagnostics();
    
    // Step 5: Find temperature
    printf("\nSTEP 5: Finding temperature...\n");
    float good_temp = find_pfold_temperature();
    
    // Done
    printf("\n========================================\n");
    printf("DONE: Use temperature = %.2f\n", good_temp);
    printf("========================================\n");
    
    return 0;
}