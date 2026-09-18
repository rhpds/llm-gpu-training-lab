# Module 02 — Loading Pre-trained Models

### Brief Overview

Students load BERT and GPT-2 from the pre-staged Hugging Face model cache into GPU memory and measure the memory footprint of each model before fine-tuning begins. Because no internet access to Hugging Face is available in the lab environment, all model weights were pre-downloaded and cached to a shared volume by Ansible automation before students arrived. This module establishes a concrete, measurable understanding of how transformer model size maps to GPU memory consumption — a key concept for customer conversations about AI workload sizing.

### Audience and Time

- **Target personas:** Tech Sales — solution architects, technical account managers, field engineers
- **Prerequisites for this module:** Completion of Module 1 (CUDA verified, GPU tensor operations confirmed working)
- **Estimated duration:** 30 minutes

### Learning Objectives

- Load BERT and GPT-2 tokenizers and model weights from the local Hugging Face cache without internet access
- Move loaded models to GPU and measure the resulting GPU memory footprint for each model
- Compare GPU memory consumption between BERT (encoder-only) and GPT-2 (decoder-only) architectures
- Implement the load-use-free pattern to avoid exhausting GPU memory when working with multiple models

### Lab Structure

| Section | Title | Duration |
|---------|-------|----------|
| 1 | Exploring the Pre-staged Model Cache | 5 min |
| 2 | Loading and Placing BERT on GPU | 12 min |
| 3 | Loading and Placing GPT-2 on GPU | 8 min |
| 4 | Comparing Memory Footprints | 5 min |

### Detailed Steps

**Section 1 — Exploring the Pre-staged Model Cache (5 min)**

1. Open the notebook file `02-loading-models.ipynb` from the Jupyter file browser.
2. In the first cell, run `import os; print(os.environ.get('HF_HOME'))` and observe the path pointing to the shared pre-staged volume.
3. In a shell cell (prefix with `!`), run `ls $HF_HOME/hub/` and observe the directory listing.
4. Confirm that `models--bert-base-uncased` and `models--gpt2` directories are present.
5. Run `!du -sh $HF_HOME/hub/models--*` and note the size on disk of each model cache.
6. Observe in the narrative cell that no network request to Hugging Face is made at any point — `from_pretrained()` reads entirely from the local cache when `HF_HOME` is set.

**Section 2 — Loading and Placing BERT on GPU (12 min)**

7. In the next cell, import BERT classes: `from transformers import BertTokenizer, BertForSequenceClassification`.
8. Load the tokenizer from the local cache: `tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')`.
9. Observe the output — no download progress bars appear, confirming the local cache is used.
10. Record the baseline GPU memory before loading the model: `baseline = torch.cuda.memory_allocated(0)`.
11. Load the BERT model: `model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)`.
12. Observe that the model is loaded in float32 on CPU first.
13. Move the model to GPU: `model = model.to('cuda')`.
14. Record the memory after placement: `bert_mem = torch.cuda.memory_allocated(0) - baseline`.
15. Run `print(f"BERT GPU memory: {bert_mem / 1e6:.1f} MB")` and note the result in your lab notes.
16. Confirm the model is functional by running a forward pass with dummy input tensors of shape `(1, 128)` and verifying that output logits have shape `(1, 2)`.

**Section 3 — Loading and Placing GPT-2 on GPU (8 min)**

17. Free BERT from GPU memory: `del model` then `torch.cuda.empty_cache()`.
18. Verify that GPU memory returns to near-baseline: run `torch.cuda.memory_allocated(0)`.
19. Import GPT-2 classes: `from transformers import GPT2Tokenizer, GPT2LMHeadModel`.
20. Load the GPT-2 tokenizer from the local cache: `gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')`.
21. Record the baseline GPU memory before loading the GPT-2 model.
22. Load the GPT-2 model: `gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')`.
23. Move to GPU: `gpt2_model = gpt2_model.to('cuda')`.
24. Record GPT-2 GPU memory using the same calculation as step 14.
25. Run `print(f"GPT-2 GPU memory: {gpt2_mem / 1e6:.1f} MB")` and note the result.

**Section 4 — Comparing Memory Footprints (5 min)**

26. Run the pre-written summary cell that prints a comparison table of BERT vs GPT-2 GPU memory usage.
27. Observe that BERT uses approximately 420–440 MB and GPT-2 uses approximately 480–500 MB.
28. Observe in the narrative cell that both models fit individually on a single lab GPU, but loading both simultaneously would consume nearly 1 GB — relevant for shared-GPU deployment discussions.
29. Free GPT-2: `del gpt2_model` then `torch.cuda.empty_cache()`.
30. Verify that GPU memory returns to near-baseline with `torch.cuda.memory_allocated(0)`.
31. Note the load-use-free pattern in the narrative: load a model, use it, free it before loading the next — a practical rule for GPU memory management in constrained environments.

### Key Takeaways

- Pre-staging model weights with Ansible removes the single largest lab friction point — no network dependency during the lab itself
- Hugging Face `from_pretrained()` reads automatically from the local cache when `HF_HOME` points to pre-downloaded weights
- BERT (~420–440 MB) and GPT-2 (~480–500 MB) have similar GPU memory footprints despite different architectures; larger models (LLaMA, Falcon) consume significantly more
- The load-use-free pattern (load to GPU, run inference or training, delete and empty cache) is essential when memory is constrained or when switching between models
- GPU memory consumption scales with model parameter count and precision — float32 uses twice the memory of float16/bfloat16

### Infrastructure Notes

- BERT (`bert-base-uncased`) and GPT-2 (`gpt2`) model weights are pre-downloaded and cached in a shared volume by Ansible automation before students arrive; the `HF_HOME` environment variable in the workbench pod is set to the mount path of this shared volume
- The tokenized training dataset used in Module 3 is also pre-staged in this shared volume at `/data/tokenized_dataset.pt`; students begin Module 2 directly from model loading without running tokenization themselves — dataset preparation was fully handled by Ansible automation
- No outbound network access to `huggingface.co` is available from within the lab cluster; any attempt to load a model not present in the pre-staged cache will fail with a connection error rather than a download
