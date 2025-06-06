#!/bin/bash

# Definiranje ulaznih i izlaznih direktorija
T1_DIR="../raw_data/IXI-T1"
T2_DIR="../raw_data/IXI-T2"
OUTPUT_T1_DIR="../data2/images/t1"
OUTPUT_T2_DIR="../data2/images/t2"
PROGRESS_FILE="../data2/processing_progress.txt"

# Stvaranje izlaznih direktorija ako ne postoje
mkdir -p $OUTPUT_T1_DIR
mkdir -p $OUTPUT_T2_DIR
mkdir -p ../data2/tmp  # Privremeni direktorij za međurezultate

# Postavljanje MNI template-a
MNI_BRAIN=$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz

# Kreiranje prazne progress datoteke ako ne postoji
if [ ! -f "$PROGRESS_FILE" ]; then
    touch "$PROGRESS_FILE"
    echo ">>> Stvorena nova datoteka napretka."
else
    echo ">>> Pronađena postojeća datoteka napretka. Nastavit ću gdje sam stao."
fi

# Dohvaćanje liste već obrađenih subjekata
PROCESSED_SUBJECTS=$(cat "$PROGRESS_FILE" | sort | uniq)

# Broj procesiranih subjekata i ukupan broj
PROCESSED=$(cat "$PROGRESS_FILE" | wc -l)
TOTAL=$(ls $T1_DIR/*.nii.gz | wc -l)

echo ">>> Započinjem obradu $TOTAL ispitanika (već obrađeno: $PROCESSED)..."

# Funkcija za čistu terminaciju (za hvatanje Ctrl+C)
cleanup() {
    echo -e "\n>>> Obrada prekinuta. Obrađeno $PROCESSED od $TOTAL ispitanika."
    
    # Čišćenje datoteka za trenutnog ispitanika ako postoje
    if [ -n "$SUBJECT_ID" ]; then
        echo ">>> Čistim privremene datoteke za nedovršenog ispitanika IXI$SUBJECT_ID..."
        rm -f ../data2/tmp/IXI${SUBJECT_ID}*
    fi
    
    echo ">>> Možeš nastaviti kasnije pokretanjem iste skripte."
    exit 1
}

# Postavljanje za Ctrl+C
trap cleanup SIGINT

# Pronalaženje i obrada svih T1 slika
for T1_PATH in $T1_DIR/*.nii.gz; do
    # Izdvajanje ID-a ispitanika iz T1 putanje
    FILENAME=$(basename $T1_PATH)
    SUBJECT_ID=$(echo $FILENAME | grep -o "IXI[0-9]\+" | sed 's/IXI//g')
    
    # Provjera je li subjekt već obrađen
    if echo "$PROCESSED_SUBJECTS" | grep -q "^${SUBJECT_ID}$"; then
        echo ">>> Subjekt IXI$SUBJECT_ID već obrađen, preskačem..."
        continue
    fi
    
    # Stvaranje odgovarajuće T2 putanje
    T2_PATH="$T2_DIR/$(basename $T1_PATH | sed 's/T1/T2/g')"
    
    # Provjera postoji li odgovarajuća T2 slika
    if [ ! -f "$T2_PATH" ]; then
        echo ">>> UPOZORENJE: Nije pronađena T2 slika za subjekt IXI$SUBJECT_ID, preskačem..."
        # Dodaj u listu procesiranih da ga ne pokušavamo ponovo
        echo "$SUBJECT_ID" >> "$PROGRESS_FILE"
        continue
    fi
    
    # Stvaranje prefiksa za izlazne datoteke
    OUTPUT_PREFIX="../data2/tmp/IXI${SUBJECT_ID}"
    
    echo ">>> [$((PROCESSED+1))/$TOTAL] Obrađujem subjekt IXI$SUBJECT_ID..."
    
    # 0️⃣ NOVI KORAK: Standardizacija orijentacije izvornih slika
    fslreorient2std $T1_PATH ${OUTPUT_PREFIX}_T1_reorient.nii.gz
    fslreorient2std $T2_PATH ${OUTPUT_PREFIX}_T2_reorient.nii.gz
    
    # 1️⃣ Uklanjanje lubanje iz reorientiranih slika
    bet ${OUTPUT_PREFIX}_T1_reorient.nii.gz ${OUTPUT_PREFIX}_T1_brain.nii.gz -R -f 0.6 -m
    
    # 2️⃣ Uklanjanje lubanje iz T2
    bet ${OUTPUT_PREFIX}_T2_reorient.nii.gz ${OUTPUT_PREFIX}_T2_brain.nii.gz -R -f 0.3 -m
    
    # 3️⃣ Linearna registracija T1 na MNI prostor s poboljšanim parametrima
    flirt -in ${OUTPUT_PREFIX}_T1_brain.nii.gz -ref $MNI_BRAIN -out ${OUTPUT_PREFIX}_T1_MNI.nii.gz -omat ${OUTPUT_PREFIX}_T1_to_MNI.mat -interp spline -dof 12
    
    # 4️⃣ Registracija T2 na T1
    flirt -in ${OUTPUT_PREFIX}_T2_brain.nii.gz -ref ${OUTPUT_PREFIX}_T1_brain.nii.gz -out ${OUTPUT_PREFIX}_T2_to_T1.nii.gz -omat ${OUTPUT_PREFIX}_T2_to_T1.mat -interp spline
    
    # 5️⃣ Primjena MNI transformacije na T2
    flirt -in ${OUTPUT_PREFIX}_T2_to_T1.nii.gz -ref $MNI_BRAIN -out ${OUTPUT_PREFIX}_T2_MNI.nii.gz -applyxfm -init ${OUTPUT_PREFIX}_T1_to_MNI.mat -interp spline
    
    # 6️⃣ Slice-anje iz MNI prostora - jednostavan kod koji samo sprema slike
python3 <<EOF
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# Definiraj putanje do datoteka
output_prefix = "$OUTPUT_PREFIX"
t1_path = f"{output_prefix}_T1_MNI.nii.gz"
t2_path = f"{output_prefix}_T2_MNI.nii.gz"

# Provjeri postoje li datoteke
if not os.path.exists(t1_path) or not os.path.exists(t2_path):
    print(f"UPOZORENJE: Nedostaju slike za subjekt IXI${SUBJECT_ID}")
    exit(1)

# Učitaj registrirane slike
img_T1 = nib.load(t1_path)
img_T2 = nib.load(t2_path)
data_T1 = img_T1.get_fdata()
data_T2 = img_T2.get_fdata()

# Uzmi točno sredinu volumena za konzistentnost
z_coord = data_T1.shape[2] // 2

# Izvuci aksijalni slice (koristimo [:, :, z] za aksijalni presjek)
axial_slice_T1 = data_T1[:, :, z_coord]
axial_slice_T2 = data_T2[:, :, z_coord]

# Provjera orijentacije - možda trebamo flipati/rotirati za konzistentnost
# Osiguravamo da je lijeva strana mozga prikazana na lijevoj strani slike
axial_slice_T1 = np.fliplr(axial_slice_T1)
axial_slice_T2 = np.fliplr(axial_slice_T2)

# Direktno spremanje sliceova kao slike bez prikazivanja
plt.imsave("$OUTPUT_T1_DIR/IXI${SUBJECT_ID}_T1_slice.png", axial_slice_T1.T, cmap="gray")
plt.imsave("$OUTPUT_T2_DIR/IXI${SUBJECT_ID}_T2_slice.png", axial_slice_T2.T, cmap="gray")

EOF
    
    # Provjera jesu li generirane slike
    if [ ! -f "$OUTPUT_T1_DIR/IXI${SUBJECT_ID}_T1_slice.png" ] || [ ! -f "$OUTPUT_T2_DIR/IXI${SUBJECT_ID}_T2_slice.png" ]; then
        echo "!!! GREŠKA: Neuspjela ekstrakcija sliceova za IXI$SUBJECT_ID"
        continue
    fi
    
    # Čišćenje međurezultata za uštedu prostora
    rm ${OUTPUT_PREFIX}_T1_reorient.nii.gz ${OUTPUT_PREFIX}_T2_reorient.nii.gz
    rm ${OUTPUT_PREFIX}_T1_brain.nii.gz ${OUTPUT_PREFIX}_T1_brain_mask.nii.gz
    rm ${OUTPUT_PREFIX}_T2_brain.nii.gz ${OUTPUT_PREFIX}_T2_brain_mask.nii.gz
    rm ${OUTPUT_PREFIX}_T1_MNI.nii.gz ${OUTPUT_PREFIX}_T1_to_MNI.mat
    rm ${OUTPUT_PREFIX}_T2_to_T1.nii.gz ${OUTPUT_PREFIX}_T2_to_T1.mat
    rm ${OUTPUT_PREFIX}_T2_MNI.nii.gz
    
    # Označavanje subjekta kao obrađenog
    echo "$SUBJECT_ID" >> "$PROGRESS_FILE"
    
    PROCESSED=$((PROCESSED+1))
    echo ">>> Subjekt IXI$SUBJECT_ID uspješno obrađen. ($PROCESSED/$TOTAL)"
done

echo ">>> Obrada završena! Obrađeno $PROCESSED od $TOTAL ispitanika."
echo ">>> T1 sliceovi spremljeni u $OUTPUT_T1_DIR"
echo ">>> T2 sliceovi spremljeni u $OUTPUT_T2_DIR"