# Module 01 — PyTorch GPU Setup

### Brief Overview

This module establishes the GPU-accelerated environment that all subsequent modules depend on. Students verify that the OpenShift AI-managed Jupyter notebook has CUDA access, confirm GPU device properties, and run basic PyTorch tensor operations on GPU to validate device placement and memory tracking. No model loading occurs in this module — the focus is entirely on environment and tooling validation so that students can confidently interpret GPU feedback in later modules.

### Audience and Time

- **Target personas:** Tech Sales — solution architects, technical account managers, field engineers
- **Prerequisites for this module:** Active provisioned OpenShift cluster with GPU-enabled worker node and RHOAI Jupyter workbench in Running state (provided by lab environment)
- **Estimated duration:** 25 minutes

### Learning Objectives

- Verify CUDA availability and GPU device properties within an RHOAI-managed Jupyter notebook environment
- Create and manipulate PyTorch tensors on GPU to confirm correct device placement
- Track GPU memory allocation and deallocation in real time using `torch.cuda` utilities
- Identify the GPU worker node and its resources as managed by the RHOAI operator

### Lab Structure

| Section | Title | Duration |
|---------|-------|----------|
| 1 | Accessing the Jupyter Environment | 5 min |
| 2 | Verifying CUDA and GPU Availability | 10 min |
| 3 | Basic GPU Tensor Operations | 10 min |

### Detailed Steps

**Section 1 — Accessing the Jupyter Environment (5 min)**

1. Navigate to the OpenShift AI dashboard URL provided on the lab welcome page.
2. Log in with the username and password listed in the lab environment panel.
3. Click **Data Science Projects** in the left navigation sidebar.
4. Select the pre-created data science project assigned to your lab user.
5. Click **Workbenches** and confirm that the `pytorch-gpu-lab` workbench shows a status of **Running**.
6. Click **Open** to launch the Jupyter notebook interface in a new browser tab.
7. Observe that the notebook server loads and that the default kernel is a PyTorch-ready Python kernel.

**Section 2 — Verifying CUDA and GPU Availability (10 min)**

8. In the Jupyter file browser, open the notebook file `01-gpu-setup.ipynb`.
9. In the first code cell, import PyTorch: `import torch`.
10. Run the cell and observe that no `ModuleNotFoundError` or import warning appears.
11. In the next cell, run `torch.cuda.is_available()` and observe that the output is `True`.
12. Run `torch.cuda.get_device_name(0)` and record the GPU model name in your lab notes.
13. Run `torch.cuda.get_device_properties(0)` and note the total GPU memory (in bytes) reported under `total_memory`.
14. Run `torch.cuda.memory_reserved(0)` and observe that it returns `0` — no memory has been allocated yet.
15. Note in the narrative cell that CUDA drivers, PyTorch, and all Python dependencies were pre-installed by Ansible automation before the lab began — no manual installation is required.

**Section 3 — Basic GPU Tensor Operations (10 min)**

16. In the next cell, create a CPU tensor: `x = torch.randn(1000, 1000)`.
17. Move the tensor to GPU: `x_gpu = x.to('cuda')`.
18. Run `x_gpu.device` and observe that the output shows `device(type='cuda', index=0)`.
19. Perform a matrix multiplication: `result = torch.matmul(x_gpu, x_gpu.T)`.
20. Run `torch.cuda.memory_allocated(0)` and observe that GPU memory is now non-zero.
21. Note the allocated bytes printed and consider what that represents relative to the tensor size.
22. Delete the tensors and release the cached memory: `del x_gpu, result` then `torch.cuda.empty_cache()`.
23. Re-run `torch.cuda.memory_allocated(0)` and observe that the value returns to near zero.
24. Observe in the narrative cell that GPU memory management is explicit in PyTorch — allocations do not release automatically when Python variables go out of scope without calling `empty_cache()`.

### Key Takeaways

- RHOAI pre-provisions a fully configured GPU Jupyter environment — students do not install drivers or Python packages
- `torch.cuda.is_available()` is the standard first check for confirming GPU accessibility from PyTorch
- PyTorch tensors must be explicitly moved to GPU with `.to('cuda')` — CPU tensors and GPU tensors cannot be mixed in the same operation
- GPU memory allocation is trackable in real time via `torch.cuda.memory_allocated()` — a foundational skill for debugging OOM errors in later modules
- Calling `torch.cuda.empty_cache()` releases cached but unused GPU memory back to the CUDA memory pool

### Infrastructure Notes

- The `pytorch-gpu-lab` workbench runs as an OpenShift pod scheduled onto a GPU-enabled worker node managed by the RHOAI operator
- CUDA drivers and the NVIDIA device plugin are pre-configured by Ansible automation and the RHOAI GPU operator stack before the lab begins
- GPU worker nodes carry NVIDIA device plugin tolerations applied by the RHOAI operator; the workbench pod spec includes the corresponding tolerations automatically
