import os
import shutil

# Putanje do izvora i odredišta
default_dir = "/home/matejav_diplomski/masterThesis/dataset/data_default_process/images"
fix_dir = "/home/matejav_diplomski/masterThesis/dataset/data_fix"
destination_dir = "/home/matejav_diplomski/masterThesis/dataset/data/images"

# # Osiguraj da postoje odredišni direktoriji (t1 i t2 unutar data/images)
os.makedirs(os.path.join(destination_dir, "t1"), exist_ok=True)
os.makedirs(os.path.join(destination_dir, "t2"), exist_ok=True)

# 1) Kopiraj sve slike iz default procesa
for t in ["t1", "t2"]:
    source_path = os.path.join(default_dir, t)
    dest_path = os.path.join(destination_dir, t)
    for fname in os.listdir(source_path):
        src_file = os.path.join(source_path, fname)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dest_path)

print("Slike iz default procesa su uspješno kopirane!")

# 2) Pronađi sve poddirektorije unutar fix_dir i zatim kopiraj (i prepiši) sve slike
subdirs = [d for d in os.listdir(fix_dir) if os.path.isdir(os.path.join(fix_dir, d))]

for sub in subdirs:
    for t in ["t1", "t2"]:
        fix_path = os.path.join(fix_dir, sub, "images", t)
        dest_path = os.path.join(destination_dir, t)
        # Ako neki fix poddirektorij ne sadrži odgovarajući images/t1 ili images/t2, samo preskoči
        if not os.path.isdir(fix_path):
            continue
        for fname in os.listdir(fix_path):
            src_file = os.path.join(fix_path, fname)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dest_path)

print("Slike su uspješno kopirane i prepisane!")

subjects_to_remove = ["430", "433", "462", "463"]
for t in ["t1", "t2"]:
    dir_path = os.path.join(destination_dir, t)
    for fname in os.listdir(dir_path):
        # Ako naziv datoteke sadrži jednog od neželjenih ispitanika (npr. "IXI462"), izbriši datoteku
        if any(f"IXI{sub}" in fname for sub in subjects_to_remove):
            file_to_remove = os.path.join(dir_path, fname)
            os.remove(file_to_remove)

print("Slike su uspješno izbrisane!")