import argparse
import os
import time

import torch
from PIL import Image

from optimum.rbln import RBLNQwenImage21Pipeline


# Compile once (DiT + text encoder graph banks, ~1-2 hours), then reuse the output directory:
#
#   python run_qwenimage21.py --model-id Qwen/Qwen-Image-2.1-Turbo --save-dir qwen-image-2.1-turbo-rbln --export
#   python run_qwenimage21.py --model-id qwen-image-2.1-turbo-rbln --prompt "..." --width 2048 --height 2048
#   python run_qwenimage21.py --model-id qwen-image-2.1-turbo-rbln --image in.png --prompt "Make it snowy at night."
#
# Each VAE image size is compiled the first time it is used and cached under <save-dir>/vae.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen-Image-2.1-Turbo")
    parser.add_argument("--save-dir", default="qwen-image-2.1-turbo-rbln")
    parser.add_argument("--export", action="store_true")
    parser.add_argument(
        "--prompt", default="A steaming pot of tteokbokki at a night street stall, sign reading 'RBLN 분식'."
    )
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--image", nargs="*", default=[], help="condition images for editing (up to 10)")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dit-device", default="0,1", help="device list for the DiT layer groups")
    parser.add_argument("--aux-device", type=int, default=0, help="text and vision encoders")
    parser.add_argument("--output", default="output.png")
    return parser.parse_args()


def main():
    args = parse_args()
    rbln_config = {"dit_device": [int(d) for d in args.dit_device.split(",")], "aux_device": args.aux_device}
    if args.export or not os.path.isdir(args.model_id):
        pipe = RBLNQwenImage21Pipeline.from_pretrained(
            args.model_id, export=True, model_save_dir=args.save_dir, rbln_config=rbln_config
        )
    else:
        pipe = RBLNQwenImage21Pipeline.from_pretrained(args.model_id, rbln_config=rbln_config)

    kwargs = {"prompt": args.prompt, "true_cfg_scale": args.true_cfg_scale}
    if args.negative_prompt is not None:
        kwargs["negative_prompt"] = args.negative_prompt
    if args.image:
        kwargs["image"] = [Image.open(p) for p in args.image]
    if args.width and args.height:
        kwargs["width"], kwargs["height"] = args.width, args.height

    t0 = time.perf_counter()
    image = pipe(generator=torch.Generator("cpu").manual_seed(args.seed), **kwargs).images[0]
    print(f"e2e {time.perf_counter() - t0:.2f}s -> {args.output}")
    image.save(args.output)


if __name__ == "__main__":
    main()
