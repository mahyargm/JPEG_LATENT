"""Decode reconstructed images from saved latent (.pt) files produced by the JPEG AI encoder.

Usage (run from inside jpeg_latent/):
    python decode_from_latents.py \
        --latent_dir  /path/to/latents  \
        --out_dir     /path/to/output   \
        [--model_id   0]

The latent directory must contain .pt files saved by the encoder (each holds a
concatenated y_hat tensor of shape [1, N_luma+N_chroma, H_lat, W_lat]).

Image size is inferred from latent spatial dims * alignment_size unless
--img_height / --img_width are given explicitly.
"""

import os
import sys
import argparse
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def setup_ce(device: str, models_dir_name: str):
    """Build and return a configured CodingEngine with models loaded.

    Reuses the standard RecoDecoder initialisation path so that all model
    parameters and device placement are handled identically to a normal decode.
    """
    from src.codec import get_downloader
    from src.reco.coders.decoder import RecoDecoder
    from src.codec.coders import def_decoder_base_parser, def_decoder_parser_decorator

    base_parser = def_decoder_base_parser('latent_decode')
    parser_decorator = def_decoder_parser_decorator(base_parser)
    coder = RecoDecoder(base_parser, parser_decorator)

    # Provide required positional args (files don't need to exist for init)
    dummy_args = ['dummy.bits', 'dummy_out.png', '--device', device]
    coder.init_common_codec(build_model=True, cmd_args=dummy_args)

    if device == 'gpu':
        coder.init_cuda()

    coder.ce.load_models_recursively(
        get_downloader(models_dir_name, critical_for_file_absence=True)
    )

    if device == 'gpu':
        coder.ce.cuda()

    return coder.ce


def decode_latent(ce, y_hat: torch.Tensor, out_path: str,
                  img_height: int, img_width: int,
                  image_data_bits: int = 8,
                  s_ver: int = 1, s_hor: int = 1):
    """Reconstruct and save one image from a pre-loaded y_hat tensor."""
    rec = ce.decompress_from_latent(
        y_hat,
        img_height=img_height,
        img_width=img_width,
        image_data_bits=image_data_bits,
        s_ver=s_ver,
        s_hor=s_hor,
    )
    rec.write_file(out_path, bit_depth=image_data_bits)
    print(f'  -> {out_path}')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description='Decode JPEG AI latents to images')
    ap.add_argument('--latent_dir', required=True,
                    help='Directory containing .pt latent files')
    ap.add_argument('--out_dir', required=True,
                    help='Directory to write reconstructed images')
    ap.add_argument('--device', default='gpu', choices=['cpu', 'gpu'],
                    help='Device to run on (default: gpu)')
    ap.add_argument('--models_dir_name', default='models',
                    help='Name of the models directory (default: models)')
    ap.add_argument('--img_height', type=int, default=None,
                    help='Override image height (pixels); inferred from latent by default')
    ap.add_argument('--img_width', type=int, default=None,
                    help='Override image width (pixels); inferred from latent by default')
    ap.add_argument('--output_bit_depth', type=int, default=8, choices=[8, 10],
                    help='Output image bit depth (default: 8)')
    ap.add_argument('--model_id', type=int, default=None,
                    help='Force a specific model index (overrides bpp-based selection)')
    return ap.parse_args()


def latent_to_img_size(y_hat: torch.Tensor, alignment_size: int):
    """Infer image size from latent spatial dimensions * alignment_size."""
    _, _, h_lat, w_lat = y_hat.shape
    return h_lat * alignment_size, w_lat * alignment_size


def main():
    args = parse_args()

    latent_dir = Path(args.latent_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latent_files = sorted(latent_dir.glob('*.pt'))
    if not latent_files:
        print(f'No .pt files found in {latent_dir}')
        sys.exit(1)

    print(f'Found {len(latent_files)} latent file(s) in {latent_dir}')

    device = args.device
    if device == 'gpu' and not torch.cuda.is_available():
        print('CUDA not available, falling back to CPU')
        device = 'cpu'
    if device == 'gpu':
        torch.backends.cudnn.deterministic = True

    print('Building coding engine and loading models...')
    ce = setup_ce(device, args.models_dir_name)
    print('Models loaded.\n')

    if args.model_id is not None:
        ce.model.CCS_SGMM.active_tool_idx = args.model_id
        print(f'Using model_id={args.model_id}')

    alignment = ce.model.CCS_SGMM.get_active_tool().model_y.alignment_size

    for lat_path in latent_files:
        out_path = str(out_dir / (lat_path.stem + '.png'))
        print(f'Decoding {lat_path.name} ...')

        y_hat = torch.load(str(lat_path), map_location='cpu')

        if args.img_height is not None and args.img_width is not None:
            img_h, img_w = args.img_height, args.img_width
        else:
            img_h, img_w = latent_to_img_size(y_hat, alignment)

        if device == 'gpu':
            y_hat = y_hat.cuda()

        decode_latent(ce, y_hat, out_path,
                      img_height=img_h,
                      img_width=img_w,
                      image_data_bits=args.output_bit_depth)

    print(f'\nDone. {len(latent_files)} image(s) written to {out_dir}')


if __name__ == '__main__':
    main()
