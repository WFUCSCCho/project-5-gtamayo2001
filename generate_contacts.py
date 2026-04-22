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
CONTACT_CUTOFF = 8.0          # Angstroms - maximum distance for a contact
MIN_SEQUENCE_SEPARATION = 3   # Ignore residues closer than this in sequence
ATOM_NAME = "CA"              # Use alpha carbons (typical for proteins)
# ============================================================

def read_pdb_ca_coords(pdb_file):
    """
    Extract CA atom coordinates from PDB file.
    Returns: list of (resid, x, y, z)
    """
    coords = []
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_contacts.py <native.pdb> [output_file]")
        print("Example: python generate_contacts.py folded.pdb contacts.dat")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "contacts.dat"
    
    print(f"Reading {pdb_file}...")
    coords = read_pdb_ca_coords(pdb_file)
    print(f"Found {len(coords)} residues with {ATOM_NAME} atoms")
    
    if len(coords) == 0:
        print(f"ERROR: No {ATOM_NAME} atoms found.")
        print("Try changing ATOM_NAME variable to something else (e.g., 'CB', 'P', or 'ALL')")
        sys.exit(1)
    
    print(f"Finding contacts (cutoff={CONTACT_CUTOFF} A, min_seq_sep={MIN_SEQUENCE_SEPARATION})...")
    contacts = generate_contacts(coords, CONTACT_CUTOFF, MIN_SEQUENCE_SEPARATION)
    
    print(f"Writing {len(contacts)} contacts to {output_file}...")
    write_contacts_file(contacts, output_file)
    
    print("Done!")
    print(f"\nFirst 5 contacts (if any):")
    for i, c in enumerate(contacts[:5]):
        print(f"  {c[0]} -- {c[1]} : {c[2]} A")
    
    if len(contacts) == 0:
        print("\nWARNING: No contacts found!")
        print("Try increasing CONTACT_CUTOFF or checking your PDB file.")

if __name__ == "__main__":
    main()