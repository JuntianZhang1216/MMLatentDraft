# Interlude

**Interlude: Reasoning Between Tokens in Vision-Language Models**

Anonymous code release accompanying the paper.

Interlude interleaves natural-language reasoning with short latent segments.
Within each segment, the vision-language model recurrently processes its own
hidden states without decoding intermediate words, then resumes text generation.
Training combines stage-wise semantic guidance through the language head with
a geometric margin objective that favors an intermediate visual–textual
reference over either modality alone.

The released implementation uses **Qwen3-VL-8B-Instruct**. Internal names such
as `NLDModel`, `NativeLatentThinker`, `rld`, and `nld` are retained in the code
and configuration for compatibility.

## Release contents

This repository includes the model implementation, training and inference
entry points, a training configuration template, and selected analysis tools.
Training data, images, model checkpoints, analysis inputs, and a complete
benchmark evaluation pipeline are not included. Running the examples requires
supplying the corresponding model and data files.

```text
.
├── rld/
│   ├── __init__.py
│   ├── model_v2.py                         # Model and latent/text execution
│   ├── latent_thinker.py                   # Recurrent latent computation
│   ├── visual_anchor.py                    # Visual–textual reference utilities
│   ├── data.py                             # Dataset and collator
│   ├── trainer_nld.py                      # Custom training loop support
│   └── inference_utils.py                  # Generation and fallback helpers
├── scripts/
│   ├── train_nld.py                        # Training entry point
│   ├── inference.py                        # Single-image and batch inference
│   ├── start_training_stage1.sh
│   ├── start_training_stage2.sh
│   └── start_inference.sh
├── configs/
│   ├── nld_train.yaml                      # Stage 2 configuration template
│   └── fsdp_config.json
├── utils/
│   ├── analyze_efficiency.py
│   ├── analyze_latent_distribution.py
│   ├── visualize_modality_manifold.py
│   └── orthogonal_decomposition_analysis.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Download and extract the anonymous repository archive, then run all commands
below from its root directory.

Use a CUDA environment with a compatible PyTorch and FlashAttention 2
installation. The dependency file pins `transformers==5.3.0`; its other
version constraints are not a fully locked or independently validated
environment specification.

```bash
python -m pip install -r requirements.txt
```

The plotting utilities additionally require Matplotlib:

```bash
python -m pip install matplotlib
```

Obtain the Qwen3-VL-8B-Instruct model and processor files separately. Pass
their local directory explicitly in the configuration or inference command.

## Training data

The training loader reads a JSON array. Each record contains image paths,
a question, an answer, and `reasoning_for_training`. The following is an
illustrative schema example, not a released training sample:

```json
[
  {
    "image": "images/example.jpg",
    "question": "Which object is closer to the camera?",
    "answer": "The red cube.",
    "reasoning_for_training": "I compare the objects' depth cues. <|latent|><|/latent|> The red cube appears closer.",
    "latent_key_tokens": [
      [
        {"tokens": ["depth", "occlusion"], "role": "concrete"},
        {"tokens": ["relative distance"], "role": "bridge"}
      ]
    ]
  }
]
```

- `image` or `image_path` specifies a single image; `image_paths` accepts a
  list for multi-image training samples. Relative paths are resolved against
  `data.image_base_dir`.
- `reasoning_for_training` contains the explicit trace and latent boundaries.
  The loader appends the `Final Answer:` section from `answer`.
- `latent_key_tokens` has the structure `[boundary][stage]`. Its outer list
  must align with the `<|latent|>` occurrences in the reasoning trace. Each
  inner list determines that boundary's number of latent training steps.
- Each stage provides concept strings in `tokens` and a `role`:
  `abstract`, `bridge`, `unified`, or `concrete`.

The loader also accepts legacy `<|pause|>` markers and converts them during
preprocessing. Annotation-generation scripts are not part of this release.

## Training

Copy the supplied template before editing it:

```bash
cp configs/nld_train.yaml configs/local_train.yaml
```

Set the following fields in `configs/local_train.yaml`:

| Field | Value to provide |
| --- | --- |
| `model.model_path` | Local Qwen3-VL-8B-Instruct directory |
| `data.train_json` | Training JSON file |
| `data.image_base_dir` | Base directory for relative image paths |
| `nld.resume_from_model_only` | Compatible previous-stage checkpoint directory, or `null` to initialize from the base model |
| `training.output_dir` | Directory for checkpoints and logs |

For model-only initialization, the previous-stage directory must contain
compatible `.safetensors` model weights. Verify that it exists: the entry
point can skip loading when the supplied path does not exist.

Launch distributed training, adjusting GPU selection and process count to
your hardware:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 scripts/train_nld.py \
  --config configs/local_train.yaml
```

The template enables FSDP and bfloat16. GPU memory requirements depend on image
resolution, sequence length, and batch settings. Final model export is written
to `<training.output_dir>/model`.

To resume a compatible Trainer checkpoint, append
`--resume_from_checkpoint /path/to/checkpoint` to the training command.
Clear `nld.resume_from_model_only` when using this mode to avoid a separate
model-only initialization.

The repository contains two stage-named shell launchers, but only one Stage 2
YAML template. The launcher name does not select a training stage; stage
behavior depends on the supplied data and configuration. A separate,
paper-matched Stage 1 recipe is not included.

## Inference

Use the Python entry point with both the base model and a trained Interlude
checkpoint:

```bash
python scripts/inference.py \
  --model_path /path/to/Qwen3-VL-8B-Instruct \
  --nld_checkpoint /path/to/training_output/model \
  --image /path/to/image.jpg \
  --question "What is shown in this image?" \
  --max_new_tokens 512 \
  --temperature 0.7 \
  --device cuda
```

For batch inference, create a JSON array such as:

```json
[
  {
    "image": "/path/to/image.jpg",
    "question": "What is shown in this image?"
  }
]
```

Then run:

```bash
python scripts/inference.py \
  --model_path /path/to/Qwen3-VL-8B-Instruct \
  --nld_checkpoint /path/to/training_output/model \
  --batch_file /path/to/queries.json \
  --output_file results.json
```

The batch entry point processes one image per query. Its default generation
uses sampling and enables a second, direct-answer attempt if the first output
hits the generation limit, lacks `Final Answer:`, or triggers the repetition
check. These defaults should be accounted for when comparing evaluation
results. The CLI does not expose every model or generation setting.

Use the direct Python and `torchrun` commands above: the bundled inference
launcher and the default Stage 2 launcher configuration path contain directory
assumptions that do not match this repository layout.

## Analysis tools

The scripts under `utils/` cover efficiency measurements, latent-distribution
plots, modality diagnostics, and orthogonal-decomposition analysis. Inspect
each script's arguments and expected input format before running it; their
required checkpoints, logs, or intermediate results must be supplied
separately. These tools do not constitute a complete reproduction pipeline
for every paper table or figure.

## Special tokens

| Token | Purpose |
| --- | --- |
| `<|latent|>` | Begin a latent reasoning segment |
| `<|/latent|>` | End a latent reasoning segment |
| `<|pause|>` | Legacy data marker converted during preprocessing |
