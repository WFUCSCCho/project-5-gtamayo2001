#!/usr/bin/env python3
"""
Generate native contacts list from a PDB file for coarse-grained Go-model.
Assumes one bead per residue (CA atoms or backbone centroid).
"""

import sys
import math
from pathlib import Path

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
                    res_id = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((res_id, x, y, z))
    return coords

def calculate_distance(coord1, coord2):
    """Euclidean distance between two 3D points."""
    _, x1, y1, z1 = coord1
    _, x2, y2, z2 = coord2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

def generate_contacts(coords, cutoff, min_sep):
    """
    Find all residue pairs within cutoff distance.
    Returns: list of (i, j, distance)
    """
    contacts = []
    n = len(coords)
    
    for i in range(n):
        for j in range(i + min_sep, n):  # Skip neighbors in sequence
            dist = calculate_distance(coords[i], coords[j])
            if dist <= cutoff:
                # Store as 1-indexed for typical MD conventions
                contacts.append((i+1, j+1, round(dist, 2)))
    
    return contacts

def write_contacts_file(contacts, output_file):
    """Write contacts in simple format."""
    with open(output_file, 'w') as f:
        f.write(f"# Native contacts generated from PDB\n")
        f.write(f"# Total contacts: {len(contacts)}\n")
        f.write(f"# Format: bead_i  bead_j  native_distance\n")
        f.write(f"{len(contacts)}\n")
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
        print(f"ERROR: No {ATOM_NAME} atoms found. Try changing ATOM_NAME variable.")
        sys.exit(1)
    
    print(f"Finding contacts (cutoff={CONTACT_CUTOFF}Å, min_seq_sep={MIN_SEQUENCE_SEPARATION})...")
    contacts = generate_contacts(coords, CONTACT_CUTOFF, MIN_SEQUENCE_SEPARATION)
    
    print(f"Writing {len(contacts)} contacts to {output_file}...")
    write_contacts_file(contacts, output_file)
    
    print("Done!")
    print(f"\nFirst 5 contacts:")
    for c in contacts[:5]:
        print(f"  {c[0]} -- {c[1]} : {c[2]} Å")

if __name__ == "__main__":
    main()
