# HOGWOOF

A state-of-the-art image compression algorithm that achieves absurd compression ratios by deleting almost all of the image.

HOGWOOF works in two stages:

```text
image → caption → tiny file
tiny file → diffusion model → plausible replacement image
```

During encoding, a vision-language model describes the input image in a short text prompt. The original pixels are then discarded.

During decoding, a generative image model reads that description and creates a new image that is semantically similar to the original, though almost certainly not the same photograph.

## Example

```bash
python hogwoof.py encode family.jpg family.hog
python hogwoof.py decode family.hog remembered-family.png
```

A `.hog` file might contain:

```json
{"v":1,"w":4032,"h":3024,"p":"a family sitting together on a couch"}
```

Congratulations: your 20 MB photograph is now approximately 100 bytes.

## Is this actually compression?

Technically: yes, if you are sufficiently irresponsible with the word *compression*.

More precisely, HOGWOOF is a **semantic lossy codec relative to a shared generative prior**. Most of the information required to reconstruct a plausible image lives in the decoder model rather than in the compressed file itself.

In other words:

```text
compressed image = vague memory + several gigabytes of pretrained neural network
```

## Requirements

```bash
pip install torch transformers diffusers accelerate pillow sentencepiece
```

A GPU is strongly recommended.

## Warning

Do **not** use HOGWOOF for:

- family archives
- legal evidence
- scientific images
- medical imaging
- anything you ever want to see again exactly as it was

HOGWOOF does not preserve photographs.

It preserves **the general idea of photographs**.
