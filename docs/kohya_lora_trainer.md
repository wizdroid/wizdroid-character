# Kohya LoRA Trainer Implementation Plan

_Last updated: 2025-11-15_

## 1. Goal & Scope
- Enable repeatable LoRA training for Wizdroid character prompts so artists can fine-tune SD/SDXL-style checkpoints on curated character sets.
- Leverage the community-standard **kohya_ss** scripts for training, scheduling, and resume support.
- Produce LoRA weights plus metadata files that plug back into existing ComfyUI workflows and the `character_prompt_node` outputs.

## 2. Success Criteria
1. **Train**: Run kohya `train_network.py` with reproducible configs for at least one demo dataset.
2. **Traceability**: Capture dataset provenance, hyper-parameters, and git hashes in an experiment log.
3. **Deployment**: Export `.safetensors` LoRA weights and register them inside ComfyUI (e.g., `/models/lora/`).
4. **Handoff**: Document a soup-to-nuts flow (data prep → training → validation → upload) that a teammate can follow without reverse-engineering.

## 3. Proposed Repository Additions
| Path | Purpose |
| --- | --- |
| `training/kohya/README.md` | Quickstart for running the trainer plus troubleshooting notes |
| `training/kohya/configs/*.yaml` | Versioned kohya config templates (SD 1.5, SDXL) |
| `training/kohya/datasets/` | Symlinked or placeholder folder for curated image datasets |
| `training/kohya/prompts/*.jsonl` | Prompt captions exported from the Character Prompt Builder |
| `scripts/train_lora.sh` | Thin wrapper around kohya CLI with env checks and logging |
| `docs/kohya_lora_trainer.md` | This document (authoritative plan + SOP) |

> _Assumption_: kohya_ss repo is vendored as a submodule under `third_party/kohya_ss/` to avoid mutation of upstream scripts while still allowing updates.

## 4. Dependencies & Environment
- **Python**: 3.10+ (matches current `.venv`).
- **CUDA Toolkit**: 12.x w/ cuDNN 9; confirm compatibility with existing GPU nodes.
- **PyTorch**: Mirror ComfyUI build (torch 2.1+ w/ CUDA 12). Keep versions pinned in `requirements.txt` or a new `training/requirements-training.txt`.
- **kohya_ss**: Specific commit hash recorded in `training/kohya/VERSION` to guarantee determinism.
- **Extras**: bitsandbytes (optional), xformers, accelerate, tensorboard, wandb (optional but recommended for tracking).

### Environment Setup Steps
1. Create a new optional virtualenv (`.venv-kohya`) to isolate heavy training deps.
2. Install kohya requirements using `pip install -r kohya_ss/requirements.txt` (or curated equivalents).
3. Run a quick GPU smoke test (`python kohya_ss/train_network.py --help`) to confirm compiled extensions.

## 5. Data Preparation Pipeline
1. **Character export**: Extend `character_prompt_node.py` with a silent "Export Caption" toggle that saves prompts (+ negative prompts) to `training/kohya/prompts/{character_id}.jsonl`.
2. **Image curation**: Collect 20–200 high-quality renders per character; enforce consistent aspect ratios (Photo Aspect Extractor can validate).
3. **Caption alignment**: Use BLIP/LLava auto-captioning only as a fallback—prefer curated captions from prompts for consistency.
4. **Directory schema**:
   ```
   datasets/
     character_slug/
       images/*.png
       captions.jsonl   # {"file_name": "0001.png", "prompt": "..."}
       metadata.yaml    # character notes, licensing, pose split
   ```
5. **Quality gates**: Run a lint script to ensure no missing captions, enforce min resolution (768px shortest side), and detect NSFW tags for rating compliance.

## 6. Training Workflow (Step-by-Step)
1. **Select template**: Copy `configs/sdxl_character.yaml` → `configs/<character>.yaml` and fill dataset paths + hyperparams.
2. **Token counting**: Use kohya `dataset_config.toml` to map triggers (e.g., `"wizchar_A"`). Document in metadata.
3. **Kickoff script**: Execute `scripts/train_lora.sh configs/<character>.yaml`.
4. **Logging**: Pipe stdout/stderr to `training/logs/<character>/<timestamp>.log`. Optionally stream to TensorBoard/W&B.
5. **Checkpoint cadence**: Save every 100–200 steps plus best-loss symlink.
6. **Validation**: After each checkpoint, run an automated ComfyUI batch using the Character Prompt Builder + LoRA to render a 2x2 grid. Store outputs under `training/validations/`.
7. **Early stopping**: Monitor Kohya `losses.json`; stop when overfitting triggers (validation quality drop) or target PSNR reached.
8. **Packaging**: Copy best `.safetensors` into `/models/lora/<character>/`. Generate a `model-card.md` with prompt tokens, training data summary, and licensing.

## 7. Integration with Existing Nodes
- **Prompt export hook**: Add optional dataset tagging UI field so prompts know the target LoRA class token.
- **Inference defaults**: Update `character_options.json` with new `lora_trigger` entries keyed by character archetype.
- **Workflow snippets**: Provide ComfyUI JSON examples showing Character Prompt Builder → LoRA loader → Sampler for consistent characters.

## 7.1 Tags (LoRA triggers) — What do we mean by a "tag"?

- A **tag** (sometimes called a trigger token) is a short, unique text token that becomes part of a prompt and signals the LoRA (or textual-inversion embedding) to produce a specific, learned characteristic. During training you label every image/prompt pair with the same tag so the model learns to associate visuals and styles with that token.

- Example tag usage in training captions:

  ```json
  {"file_name": "0001.png", "prompt": "<char_alice> wearing a vintage red dress, close-up portrait, cinematic lighting"}
  {"file_name": "0002.png", "prompt": "<char_alice> mid-twirl in a flowing red dress, studio white backdrop"}
  ```

- Example in `kohya_ss` templates: you might map a trigger to a dataset in `dataset_config.toml` like `"char_alice"` or simply ensure each caption contains your chosen token.

- Inference: once a LoRA is trained with that tag, prompts using the same tag will activate the learned characteristics. For example, adding `"<char_alice>"` to your ComfyUI prompt should bias generation toward the trained character. If you use the ComfyUI LoRA loader instead, you can also apply the `.safetensors` weight directly with a multiplier (e.g., `<lora:character_alice:0.7>`) — the loader and tag can both be used for convenience.

### Naming guidelines for tags

- Keep tags short and unique: `char_alex`, `wizchar_A`, `nurse_maria`.
- Use only letters, numbers and underscores. Avoid spaces and punctuation that can break tokenization.
- Prefer lower-case for consistency.
- One tag per character or concept; avoid re-using tags across multiple characters.

### How to add a tag to the repo and prompt export

1. **Choose a tag for the character** — e.g., `char_alice`.
2. **Add a mapping** to `character_options.json` (example snippet):

   ```json
   "lora_trigger": {
     "alice": "char_alice",
     "herbalist": "char_herbalist"
   }
   ```

   This is only a suggested pattern: it allows the `character_prompt_node` to offer a drop-down that inserts the tag token into exported prompts.

3. **Export prompts with the tag** via the `character_prompt_node`'s caption export toggle or the dataset pipeline. Ensure every image associated with `char_alice` includes the tag somewhere in its prompt.
4. **Train** with kohya as usual. The tag becomes the anchor the LoRA learns to follow.

### Example dataset manifest using tags

```
datasets/
  char_alice/
    images/0001.png
    images/0002.png
    captions.jsonl  # each entry includes "<char_alice>"
    metadata.yaml   # lora_tag: char_alice
```

### Note about LoRA loader vs tags

- Tags are textual—useful for prompt-driven workflows and prompting consistency.
- `.safetensors` LoRA weights can be applied directly with the ComfyUI LoRA loader; a tag helps you keep the dataset and prompts connected to the correct weights and provides a shorthand you can use in batches.


## 8. Automation & Scheduling
- Use `Makefile` targets (`make train-lora CHARACTER=herbalist`) to standardize runs.
- Support resumable jobs: wrappers detect existing checkpoints and call kohya with `--resume`.
- Nightly CI (optional) to lint datasets and ensure configs stay in sync with JSON catalogs.

## 9. Validation & QA
1. **Technical**: `python -m compileall` + `pytest` (future tests) before training to ensure nodes still load.
2. **Visual**: Maintain a `validation_manifest.json` storing prompt seeds + evaluation notes for each checkpoint.
3. **Regression**: When updating nodes, rerun a reference ComfyUI workflow with and without LoRA to ensure prompts still align.

## 10. Risks & Mitigations
| Risk | Impact | Mitigation |
| --- | --- | --- |
| GPU memory shortfall | Training fails mid-run | Provide low-memory config (rank=8, network_dim=32) + gradient checkpointing |
| Dataset drift | Outputs lose character identity | Version datasets, store hashes, and freeze once a LoRA is published |
| Script divergence | kohya upstream changes break automation | Pin commit hash; run weekly `git submodule update --remote --dry-run` review |
| Licensing concerns | Cannot redistribute LoRA | Require metadata.yaml to include license + release approval |

## 11. Timeline (indicative)
1. **Week 1**: Wire up repo structure, submodule, and wrapper scripts.
2. **Week 2**: Build prompt export + dataset validation tooling.
3. **Week 3**: Train first SD 1.5 LoRA, iterate on hyperparams, document process.
4. **Week 4**: Add SDXL template, publish guide, and integrate triggers into Character nodes.

## 12. Next Steps Checklist
- [ ] Add kohya_ss as a git submodule under `third_party/`.
- [ ] Scaffold `training/kohya` folder with config templates + sample dataset manifest.
- [ ] Implement prompt export toggle in `character_prompt_node.py`.
- [ ] Write `scripts/train_lora.sh` with environment sanity checks.
- [ ] Prepare demo dataset + run pilot training.
- [ ] Publish example ComfyUI workflow referencing the trained LoRA.

## 13. References
- [kohya_ss wiki](https://github.com/kohya-ss/sd-scripts/wiki)
- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [ComfyUI LoRA loader docs](https://comfyui.gitbook.io/comfyui-docs/)
