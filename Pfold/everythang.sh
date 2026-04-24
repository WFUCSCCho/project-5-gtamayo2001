#!/bin/bash

# variables
NATIVEPDB=1r69.adjusted.pdb
TPDB=1r69_T_99.pdb

# commands
python3 generate_contacts.py NATIVEPDB contacts.dat
./runme TPDB contacts.dat

