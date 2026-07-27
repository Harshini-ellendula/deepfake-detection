import os
import shutil
import random

# REAL DATASET PATH
real_source = "audio_dataset/for-original/for-original/training/real"

# FAKE DATASET ROOT FOLDERS
fake_roots = [
    "audio_dataset/for-norm",
    "audio_dataset/for-2sec",
    "audio_dataset/for-rerec"
]

target_real = "audio_dataset/real"
target_fake = "audio_dataset/fake"

os.makedirs(target_real, exist_ok=True)
os.makedirs(target_fake, exist_ok=True)

print("Collecting real files...")

real_files = []

for root, dirs, files in os.walk(real_source):
    for file in files:
        if file.endswith(".wav"):
            real_files.append(os.path.join(root, file))

selected_real = random.sample(real_files, min(800, len(real_files)))

for file in selected_real:
    shutil.copy(file, os.path.join(target_real, os.path.basename(file)))

print("Collecting fake files...")

fake_files = []

for folder in fake_roots:
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".wav"):
                fake_files.append(os.path.join(root, file))

selected_fake = random.sample(fake_files, min(800, len(fake_files)))

for file in selected_fake:
    shutil.copy(file, os.path.join(target_fake, os.path.basename(file)))

print("✅ Dataset prepared successfully")
print("Real files:", len(selected_real))
print("Fake files:", len(selected_fake))