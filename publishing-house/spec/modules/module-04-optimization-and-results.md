# Module 04 — Optimization and Results

### Brief Overview

Students profile GPU memory usage, apply gradient accumulation to simulate a larger effective batch size without increasing GPU memory consumption, and work through a guided CUDA out-of-memory error scenario from diagnosis to resolution. This module ties together the memory optimization theme that runs through the lab and gives technical sellers a concrete vocabulary — and a lived experience — for customer conversations about GPU workload efficiency and sizing on OpenShift AI.

### Audience and Time

- **Target personas:** Tech Sales — solution architects, technical account managers, field engineers
- **Prerequisites for this module:** Completion of Module 3 (fine-tuning training loop running with AMP, peak GPU memory recorded)
- **Estimated duration:** 20 minutes

### Learning Objectives

- Profile GPU memory allocation across training phases and identify which phase produces peak memory consumption
- Implement gradient accumulation to increase the effective batch size without increasing GPU memory usage
- Diagnose a CUDA out-of-memory error from its traceback and resolve it using standard PyTorch mitigation techniques
- Summarize GPU utilization and memory efficiency results across the full lab and relate them to OpenShift AI workload sizing

### Lab Structure

| Section | Title | Duration |
|---------|-------|----------|
| 1 | Memory Profiling | 6 min |
| 2 | Gradient Accumulation | 7 min |
| 3 | CUDA OOM Diagnosis and Resolution | 7 min |

### Detailed Steps

**Section 1 — Memory Profiling (6 min)**

1. Open the notebook file `04-optimization.ipynb` from the Jupyter file browser.
2. In the first cell, reset peak memory statistics to get a clean baseline: `torch.cuda.reset_peak_memory_stats(0)`.
3. Reload BERT to GPU (same call as Module 3, step 10): `model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to('cuda')`.
4. Record the model's static memory footprint: `model_mem = torch.cuda.memory_allocated(0); print(f"Model footprint: {model_mem/1e6:.1f} MB")`.
5. Run a single training step (one forward pass and one backward pass) using the AMP setup from Module 3.
6. Immediately after the backward pass, record peak memory: `peak = torch.cuda.max_memory_allocated(0); print(f"Peak training memory: {peak/1e6:.1f} MB")`.
7. Calculate the activation overhead: `print(f"Activation overhead: {(peak - model_mem)/1e6:.1f} MB")`.
8. Observe in the narrative cell that activations generated during the forward pass — not the model weights — typically account for the largest share of peak GPU memory during training.

**Section 2 — Gradient Accumulation (7 min)**

9. Set the accumulation step count: `accumulation_steps = 4`.
10. In the pre-written accumulation cell, locate the modified training loop.
11. Observe that the loss is scaled before the backward pass: `loss = loss / accumulation_steps`.
12. Observe that `optimizer.step()` and `scaler.update()` are called only when `(step + 1) % accumulation_steps == 0`.
13. Run four consecutive steps and observe in `torch.cuda.memory_allocated(0)` printouts that memory stays flat across those four steps — gradients are accumulating but no parameter update occurs.
14. Observe that `optimizer.step()` fires on the fourth step and that `torch.cuda.memory_allocated(0)` decreases slightly after the update as intermediate buffers are released.
15. Run the summary cell: `print(f"Batch size: 16 | Accumulation steps: {accumulation_steps} | Effective batch size: {16 * accumulation_steps}")`.
16. Observe that the effective batch size is 64 — equivalent to a `batch_size=64` run — while GPU memory usage remains at the level of `batch_size=16`.

**Section 3 — CUDA OOM Diagnosis and Resolution (7 min)**

17. Locate the pre-written OOM trigger cell in the notebook — it is clearly labeled with a warning comment.
18. Read the cell before running it: it sets `batch_size=512` and attempts a forward pass with that batch size.
19. Run the cell and observe the `torch.cuda.OutOfMemoryError` traceback in the output.
20. Read the error message carefully and locate the line reporting requested vs. available memory (e.g., `Tried to allocate X GiB ... only Y GiB free`).
21. Record the requested and available memory amounts in your lab notes.
22. Clear the GPU state: `torch.cuda.empty_cache()` and verify that `torch.cuda.memory_allocated(0)` returns to near the model's static footprint.
23. Apply the first fix — reduce batch size: set `batch_size=16` in the cell, re-run, and observe that the forward pass succeeds.
24. Observe the narrative cell summarizing three strategies for resolving CUDA OOM errors in order of effort: (1) reduce batch size, (2) enable or increase gradient accumulation, (3) enable AMP to halve activation memory.
25. Run the final summary cell that consolidates results from all four modules: model memory footprints (Module 2), peak training memory (Module 3), AMP memory savings, gradient accumulation effective batch size, and the OOM threshold batch size.
26. Observe that the full fine-tuning workflow — from GPU verification through training and optimization — completed on a single GPU node managed by RHOAI.

### Key Takeaways

- Peak GPU memory during training is dominated by forward-pass activations, not model weights — understanding this distinction is critical for workload sizing conversations
- Gradient accumulation allows training with a larger effective batch size on memory-constrained hardware, with no additional GPU memory cost beyond the single-batch footprint
- CUDA OOM errors are recoverable: clear the cache, reduce batch size, and retry — the model weights are not lost unless the pod crashes
- The three primary GPU memory optimization levers in PyTorch are: automatic mixed precision (AMP), gradient accumulation, and explicit cache management with `torch.cuda.empty_cache()`
- Technical sellers can now describe the full fine-tuning lifecycle — environment setup, model loading, training with AMP, memory profiling, and OOM resolution — with reference to a hands-on experience on OpenShift AI

### Infrastructure Notes

- The CUDA OOM scenario in Section 3 is designed to be deterministic at `batch_size=512` given the GPU memory available in the lab environment; if the lab GPU nodes are upgraded with higher-VRAM GPUs, the OOM trigger batch size in the notebook cell may need to be adjusted accordingly
- Peak GPU memory values reported in this module may vary by a small margin depending on the CUDA driver version and specific GPU architecture present on the cluster's worker nodes
- The final summary cell draws on `torch.cuda.max_memory_allocated()` values collected across the session — students must run the modules in order for the summary values to reflect the full training run
