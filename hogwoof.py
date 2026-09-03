#!/usr/bin/env python3

"""
HOGWOOF Image Codec
===================

Compress photographs to hundreds of bytes by discarding the photograph.

Encode:
    image -> vision-language model -> textual description -> .hog file

Decode:
    .hog -> diffusion model -> entirely new image vaguely resembling original

This is not conventional compression.
It is semantic lossy reconstruction using a gigantic external prior.

Usage:

    python hogwoof.py encode family.jpg family.hog
    python hogwoof.py decode family.hog regenerated.png

Dependencies:

    pip install torch transformers diffusers accelerate pillow sentencepiece
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import AutoPipelineForText2Image


CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
GENERATION_MODEL = "stabilityai/sdxl-turbo"


# ----------------------------------------------------------------------
# "Compression"
# ----------------------------------------------------------------------

def load_captioner():
    print(f"Loading semantic compressor: {CAPTION_MODEL}")

    processor = BlipProcessor.from_pretrained(CAPTION_MODEL)

    model = BlipForConditionalGeneration.from_pretrained(
        CAPTION_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return processor, model, device


def describe_image(image_path: Path) -> str:
    processor, model, device = load_captioner()

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=5,
        )

    description = processor.decode(
        output[0],
        skip_special_tokens=True,
    )

    return description


def encode(input_path: Path, output_path: Path):
    image = Image.open(input_path)

    prompt = describe_image(input_path)

    # Technically we could save only `prompt`.
    # A little metadata makes the format less horrifying.
    compressed = {
        "v": 1,
        "w": image.width,
        "h": image.height,
        "p": prompt,
    }

    output_path.write_text(
        json.dumps(compressed, separators=(",", ":")),
        encoding="utf-8",
    )

    original_size = input_path.stat().st_size
    compressed_size = output_path.stat().st_size

    print()
    print("Original:")
    print(f"  {input_path}")
    print(f"  {original_size:,} bytes")

    print()
    print("Compressed:")
    print(f"  {output_path}")
    print(f"  {compressed_size:,} bytes")

    print()
    print(f"Ratio: {original_size / compressed_size:,.1f}:1")

    print()
    print("Recovered semantic essence:")
    print(f'  "{prompt}"')


# ----------------------------------------------------------------------
# "Decompression"
# ----------------------------------------------------------------------

def load_generator():
    print(f"Loading decompressor: {GENERATION_MODEL}")

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    pipe = AutoPipelineForText2Image.from_pretrained(
        GENERATION_MODEL,
        torch_dtype=dtype,
        variant="fp16" if torch.cuda.is_available() else None,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)

    return pipe


def decode(input_path: Path, output_path: Path):
    data = json.loads(input_path.read_text(encoding="utf-8"))

    prompt = data["p"]
    original_width = data["w"]
    original_height = data["h"]

    pipe = load_generator()

    # Diffusion models generally want dimensions divisible by 8.
    # Also don't blindly reproduce a gigantic original image.
    max_dim = 1024

    scale = min(
        1.0,
        max_dim / max(original_width, original_height),
    )

    width = max(256, int(original_width * scale) // 8 * 8)
    height = max(256, int(original_height * scale) // 8 * 8)

    print()
    print("Consulting latent space about what your photograph")
    print("probably looked like...")
    print()
    print(f'Prompt: "{prompt}"')
    print()

    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=4,
        guidance_scale=0.0,
    ).images[0]

    image.save(output_path)

    print(f"Successfully remembered photograph as: {output_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compress images by forgetting almost everything about them."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser(
        "encode",
        help="Forget photograph and retain vague memory.",
    )
    enc.add_argument("input", type=Path)
    enc.add_argument("output", type=Path)

    dec = sub.add_parser(
        "decode",
        help="Hallucinate photograph from vague memory.",
    )
    dec.add_argument("input", type=Path)
    dec.add_argument("output", type=Path)

    args = parser.parse_args()

    if args.command == "encode":
        encode(args.input, args.output)

    elif args.command == "decode":
        decode(args.input, args.output)


if __name__ == "__main__":
    main()

