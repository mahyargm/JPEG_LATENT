import os
import shutil
import glob
from tqdm import tqdm
from collections import defaultdict

num_chunks = 5  # number of python processes

bit_output_dir = "/workspace/results/bit"
rec_output_dir = "/workspace/results/rec"
latent_output_dir = "/workspace/results/latent"
rec_dec_output_dir = "/workspace/results/rec_dec"

if not os.path.exists(bit_output_dir):
    os.makedirs(bit_output_dir)
if not os.path.exists(rec_output_dir):
    os.makedirs(rec_output_dir)
if not os.path.exists(latent_output_dir):
    os.makedirs(latent_output_dir)
if not os.path.exists(rec_dec_output_dir):
    os.makedirs(rec_dec_output_dir)

completed_files = defaultdict(set)
for i in range(num_chunks):  
    bit_path = f"/workspace/results_chunks/{i}/bit" # chunck dir path: {results_dir}/{i}/bit
    rec_path = f"/workspace/results_chunks/{i}/rec" # chunck dir path: {results_dir}/{i}/rec
    rec_dec_path = f"/workspace/results_chunks/{i}/rec_dec" # chunck dir path: {results_dir}/{i}/rec_dec
    latent_path = f"/workspace/results_chunks/{i}/latent"  # chunck dir path: {results_dir}/{i}/latent


    for file in tqdm(glob.glob(os.path.join(bit_path, "*"))):
        completed_files[os.path.basename(file).split('_')[1]].add(os.path.basename(file).split('_')[2][:-5]) 
        dest_file = os.path.join(bit_output_dir, os.path.basename(file))
        if os.path.exists(dest_file):
            os.remove(dest_file)
        shutil.move(file, bit_output_dir)

    print(f"Moved files from {bit_path} to {bit_output_dir}")

    for file in tqdm(glob.glob(os.path.join(rec_path, "*"))):
        dest_file = os.path.join(rec_output_dir, os.path.basename(file))
        if os.path.exists(dest_file):
            os.remove(dest_file)
        shutil.move(file, rec_output_dir)

    print(f"Moved files from {rec_path} to {rec_output_dir}")

    for file in tqdm(glob.glob(os.path.join(rec_dec_path, "*"))):
        dest_file = os.path.join(rec_dec_output_dir, os.path.basename(file))
        if os.path.exists(dest_file):
            os.remove(dest_file)
        shutil.move(file, rec_dec_output_dir)

    print(f"Moved files from {rec_dec_path} to {rec_dec_output_dir}")

    for file in tqdm(glob.glob(os.path.join(latent_path, "*"))):
        dest_file = os.path.join(latent_output_dir, os.path.basename(file))
        if os.path.exists(dest_file):
            os.remove(dest_file)
        shutil.move(file, latent_output_dir)

    print(f"Moved files from {latent_path} to {latent_output_dir}")

completed_images = []
for key, value in completed_files.items():
    if len(completed_files[key]) == 5:
        completed_images.append(key)

with open("/workspace/results/completed_images.txt", "w") as f:
    for image in completed_images:
        f.write(f"{image}\n")

print("All files have been moved successfully.")