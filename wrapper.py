import os
import subprocess
from multiprocessing import Pool
from pathlib import Path
import math

# Configs
EVAL_SCRIPT = "src.reco.scripts.eval"
INPUT_DIR = "/workspace/Dataset"  # Directory of images
OUTPUT_DIR = "/workspace/results"  # Output Directory
NUM_PROCESSES = 1  # Number of JPEG AI instances (Don't use numbers bigger than 6)
PYTHON_EXEC = "python"  # or path to specific python

# Get all input files
def get_image_files(input_dir):
    exts = ".png"
    return sorted([str(p) for p in Path(input_dir).rglob("*") if p.suffix.lower() == exts])

# Chunk the input list
def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

# Run one instance of the eval script on a subset of images
def run_instance(args):
    chunk_id, input_files = args
    in_chunk_dir = f"{INPUT_DIR}_chunks/{chunk_id}"
    out_chunk_dir = f"{OUTPUT_DIR}_chunks/{chunk_id}"
    os.makedirs(in_chunk_dir, exist_ok=True)
    os.makedirs(out_chunk_dir, exist_ok=True)

    # Copy input files into chunk directory (symlinks are more efficient if supported)
    for fpath in input_files:
        target = os.path.join(in_chunk_dir, os.path.basename(fpath))
        if not os.path.exists(target):
            os.symlink(fpath, target)  # or shutil.copy if symlinks not possible

    cmd = [
        PYTHON_EXEC, "-m", EVAL_SCRIPT,
        "--coding_type", "enc_dec",
        "--calc_encoder_metrics", "0",
        "--calc_decoder_metrics", "0",
        "--in_dir", in_chunk_dir,
        "--out_dir", out_chunk_dir,
        "--gpu_ids", "0",
        "--num_workers", "1",
        "--store_latent", "1"
    ]
    print(f"Launching process {chunk_id} with {len(input_files)} files")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():

    all_files = get_image_files(INPUT_DIR)

    print(f"Total files to process: {len(all_files)}")
    
    chunks = chunkify(all_files, NUM_PROCESSES)

    # Map each chunk to a separate process
    with Pool(NUM_PROCESSES) as pool:
        pool.map(run_instance, list(enumerate(chunks)))

if __name__ == "__main__":
    main()
