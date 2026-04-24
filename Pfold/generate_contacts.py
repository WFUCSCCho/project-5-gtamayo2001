#!/usr/bin/env python3
"""
Generate native contacts list from a PDB file for coarse-grained Go-model.
Assumes one bead per residue (CA atoms or backbone centroid).

OUTPUT FORMAT: Simple numbers only (no comments)
  Line 1: total_number_of_contacts
  Line 2+: bead_i bead_j native_distance

This format works directly with simple C fscanf code.
"""

import sys
import math

# ============================================================
# CONFIGURATION - ADJUST THESE TO YOUR NEEDS
# ============================================================
# STRICTER CUTOFFS to get ~60-70 contacts for folded state
CONTACT_CUTOFF = 6.5          # Angstroms - was 8.0, now stricter
MIN_SEQUENCE_SEPARATION = 4   # Ignore residues closer than this (was 3)
ATOM_NAME = "CA"              # Use alpha carbons (typical for proteins)

# Additional filtering options
USE_RELATIVE_CUTOFF = False   # Set to True to use relative cutoff
RELATIVE_CUTOFF = 1.2         # Contact if distance < 1.2 * sum of vdw radii
# ============================================================

def read_pdb_ca_coords(pdb_file):
    """
    Extract CA atom coordinates from PDB file.
    Returns: list of (resid, x, y, z)
    """
    coords = []
    seen_resids = set()  # Track residue IDs to avoid duplicates
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                if atom_name == ATOM_NAME:
                    # PDB format: residue ID is columns 23-26
                    res_id_str = line[22:26].strip()
                    if res_id_str == "":
                        continue
                    res_id = int(res_id_str)
                    
                    # Skip if we've already seen this residue (take first CA only)
                    if res_id in seen_resids:
                        continue
                    seen_resids.add(res_id)
                    
                    # Coordinates are columns 31-54
                    x_str = line[30:38].strip()
                    y_str = line[38:46].strip()
                    z_str = line[46:54].strip()
                    
                    if x_str == "" or y_str == "" or z_str == "":
                        continue
                        
                    x = float(x_str)
                    y = float(y_str)
                    z = float(z_str)
                    
                    coords.append((res_id, x, y, z))
    return coords

def calculate_distance(coord1, coord2):
    """Euclidean distance between two 3D points."""
    _, x1, y1, z1 = coord1
    _, x2, y2, z2 = coord2
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def generate_contacts(coords, cutoff, min_sep):
    """
    Find all residue pairs within cutoff distance.
    Returns: list of (i, j, distance)
    """
    contacts = []
    n = len(coords)
    
    for i in range(n):
        res_i = coords[i][0]
        for j in range(i + min_sep, n):  # Skip neighbors in sequence
            res_j = coords[j][0]
            dist = calculate_distance(coords[i], coords[j])
            if dist <= cutoff:
                # Store as 1-indexed (residue numbers as they appear in PDB)
                contacts.append((res_i, res_j, round(dist, 2)))
    
    return contacts

def generate_contacts_with_sequence_filter(coords, cutoff, min_sep, max_contacts=70):
    """
    Alternative: Try different cutoffs to hit target contact count.
    This is useful if you know the expected number of contacts.
    """
    # Try progressively stricter cutoffs until we get close to target
    test_cutoffs = [8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5]
    
    best_contacts = []
    best_cutoff = cutoff
    
    for test_cut in test_cutoffs:
        contacts = []
        for i in range(len(coords)):
            res_i = coords[i][0]
            for j in range(i + min_sep, len(coords)):
                res_j = coords[j][0]
                dist = calculate_distance(coords[i], coords[j])
                if dist <= test_cut:
                    contacts.append((res_i, res_j, round(dist, 2)))
        
        print(f"  Cutoff {test_cut}A: {len(contacts)} contacts")
        
        # Stop if we're in the right range (50-80 contacts)
        if 50 <= len(contacts) <= 80:
            print(f"  Found good cutoff: {test_cut}A gives {len(contacts)} contacts")
            return contacts, test_cut
        
        # Keep track of the best one
        if abs(len(contacts) - max_contacts) < abs(len(best_contacts) - max_contacts):
            best_contacts = contacts
            best_cutoff = test_cut
    
    print(f"  Using cutoff {best_cutoff}A with {len(best_contacts)} contacts")
    return best_contacts, best_cutoff

def write_contacts_file(contacts, output_file):
    """
    Write contacts in SIMPLE format (no comments).
    Format:
      <total_contacts>
      <bead_i> <bead_j> <distance>
      ...
    """
    with open(output_file, 'w') as f:
        # First line: total number of contacts
        f.write(f"{len(contacts)}\n")
        
        # Each following line: bead_i bead_j distance
        for i, j, dist in contacts:
            f.write(f"{i}\t{j}\t{dist}\n")

def print_statistics(contacts):
    """Print useful statistics about the contacts."""
    if len(contacts) == 0:
        return
    
    distances = [c[2] for c in contacts]
    min_dist = min(distances)
    max_dist = max(distances)
    avg_dist = sum(distances) / len(distances)
    
    print(f"\nContact Statistics:")
    print(f"  Total contacts: {len(contacts)}")
    print(f"  Distance range: {min_dist:.2f} - {max_dist:.2f} A")
    print(f"  Average distance: {avg_dist:.2f} A")
    
    # Count contacts per residue
    residue_contacts = {}
    for i, j, _ in contacts:
        residue_contacts[i] = residue_contacts.get(i, 0) + 1
        residue_contacts[j] = residue_contacts.get(j, 0) + 1
    
    if residue_contacts:
        avg_per_res = sum(residue_contacts.values()) / len(residue_contacts)
        print(f"  Average contacts per residue: {avg_per_res:.1f}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_contacts.py <native.pdb> [output_file] [--auto]")
        print("Example: python generate_contacts.py folded.pdb contacts.dat")
        print("         python generate_contacts.py folded.pdb contacts.dat --auto")
        print("\nOptions:")
        print("  --auto    Automatically find cutoff to get ~60-70 contacts")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "contacts.dat"
    auto_mode = "--auto" in sys.argv
    
    print(f"Reading {pdb_file}...")
    coords = read_pdb_ca_coords(pdb_file)
    print(f"Found {len(coords)} residues with {ATOM_NAME} atoms")
    
    if len(coords) == 0:
        print(f"ERROR: No {ATOM_NAME} atoms found.")
        print("Try changing ATOM_NAME variable to something else (e.g., 'CB', 'P', or 'ALL')")
        sys.exit(1)
    
    if auto_mode:
        print(f"\nAUTO MODE: Searching for cutoff that gives 50-80 contacts...")
        contacts, used_cutoff = generate_contacts_with_sequence_filter(
            coords, CONTACT_CUTOFF, MIN_SEQUENCE_SEPARATION, max_contacts=65
        )
        print(f"\nSelected cutoff: {used_cutoff} A")
    else:
        print(f"Finding contacts (cutoff={CONTACT_CUTOFF} A, min_seq_sep={MIN_SEQUENCE_SEPARATION})...")
        contacts = generate_contacts(coords, CONTACT_CUTOFF, MIN_SEQUENCE_SEPARATION)
    
    print(f"\nWriting {len(contacts)} contacts to {output_file}...")
    write_contacts_file(contacts, output_file)
    
    print_statistics(contacts)
    
    print(f"\nFirst 5 contacts:")
    for i, c in enumerate(contacts[:5]):
        print(f"  {c[0]} -- {c[1]} : {c[2]} A")
    
    if len(contacts) == 0:
        print("\nWARNING: No contacts found!")
        print("Try increasing CONTACT_CUTOFF or use --auto mode.")
    
    # Check if contact count matches professor's ranges
    print(f"\n========================================")
    print(f"PROFESSOR'S RANGES CHECK:")
    print(f"  Unfolded:   4-20 contacts")
    print(f"  Transition: 32-46 contacts")
    print(f"  Folded:     56-70 contacts")
    print(f"  Your folded native has: {len(contacts)} contacts")
    
    if 56 <= len(contacts) <= 70:
        print(f"  ✓ PERFECT! Your native state matches the folded range.")
    elif len(contacts) > 70:
        print(f"  ✗ TOO MANY contacts. Try lowering CONTACT_CUTOFF or use --auto")
        print(f"    Suggested: CONTACT_CUTOFF = {CONTACT_CUTOFF - 0.5}")
    elif len(contacts) < 56:
        print(f"  ✗ TOO FEW contacts. Try increasing CONTACT_CUTOFF or use --auto")
        print(f"    Suggested: CONTACT_CUTOFF = {CONTACT_CUTOFF + 0.5}")
    print(f"========================================")

if __name__ == "__main__":
    main()