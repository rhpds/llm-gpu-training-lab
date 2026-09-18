# Module 03 — Fine-tuning Workflow

### Brief Overview

This is the core module of the lab. Students implement a complete PyTorch fine-tuning training loop using BERT for sequence classification, drawing on the pre-tokenized dataset pre-staged by Ansible automation. Automatic mixed precision (AMP) is introduced as a practical optimization that reduces GPU memory consumption and increases training throughput with minimal code changes. GPU utilization is monitored live throughout the training run using `nvidia-smi`, giving students a direct visual connection between the training loop and GPU hardware behavior.

### Audience and Time

- **Target personas:** Tech Sales — solution architects, technical account managers, field engineers
- **Prerequisites for this module:** Completion of Modules 1 and 2 (GPU verified, BERT loaded on GPU at least once)
- **Estimated duration:** 45 minutes

### Learning Objectives

- Implement a complete PyTorch fine-tuning training loop for BERT sequence classification using the pre-staged tokenized dataset
- Implement automatic mixed precision (AMP) training with `torch.cuda.amp.GradScaler` to reduce memory usage and increase throughput
- Monitor GPU utilization and memory consumption in real time using `nvidia-smi` during a live training run
- Interpret training loss curves and GPU utilization graphs to assess whether training is progressing correctly

### Lab Structure

| Section | Title | Duration |
|---------|-------|----------|
| 1 | Loading the Pre-staged Dataset | 8 min |
| 2 | Configuring the Training Loop | 12 min |
| 3 | Enabling Automatic Mixed Precision | 10 min |
| 4 | Running Training and Monitoring GPU | 15 min |

### Detailed Steps

**Section 1 — Loading the Pre-staged Dataset (8 min)**

1. Open the notebook file `03-finetuning.ipynb` from the Jupyter file browser.
2. In the first cell, import required utilities: `import torch; from torch.utils.data import DataLoader, TensorDataset`.
3. Load the pre-tokenized dataset from the shared volume: `dataset = torch.load('/data/tokenized_dataset.pt')`.
4. Inspect the dataset object: `print(type(dataset)); print(dataset.keys())`.
5. Observe that the dictionary contains tensors for `input_ids`, `attention_mask`, and `labels`.
6. Construct a `TensorDataset`: `tensor_dataset = TensorDataset(dataset['input_ids'], dataset['attention_mask'], dataset['labels'])`.
7. Create a `DataLoader`: `dataloader = DataLoader(tensor_dataset, batch_size=16, shuffle=True)`.
8. Print the number of batches: `print(f"Batches per epoch: {len(dataloader)}")` and note the result.

**Section 2 — Configuring the Training Loop (12 min)**

9. Import the BERT model class: `from transformers import BertForSequenceClassification`.
10. Load and move BERT to GPU: `model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to('cuda')`.
11. Set the model to training mode: `model.train()`.
12. Configure the optimizer: `optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)`.
13. Set the number of training epochs: `num_epochs = 3`.
14. Initialize a list to record loss history: `loss_history = []`.
15. Write the outer epoch loop: `for epoch in range(num_epochs):`.
16. Inside the epoch loop, initialize a step counter: `for step, batch in enumerate(dataloader):`.
17. Unpack the batch and move all tensors to GPU in one expression: `input_ids, attention_mask, labels = [t.to('cuda') for t in batch]`.
18. Zero out gradients at the start of each step: `optimizer.zero_grad()`.
19. Run the forward pass: `outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)`.
20. Extract the loss from the model outputs: `loss = outputs.loss`.
21. Run the backward pass: `loss.backward()`.
22. Step the optimizer: `optimizer.step()`.
23. Append the loss value to the history list: `loss_history.append(loss.item())`.
24. Add a print statement that fires every 10 steps: `if step % 10 == 0: print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")`.
25. Run one epoch of this basic loop and confirm that loss values are printed and decreasing.

**Section 3 — Enabling Automatic Mixed Precision (10 min)**

26. After confirming the basic loop runs, locate the pre-written AMP variant cell in the notebook.
27. Import AMP utilities at the top of the cell: `from torch.cuda.amp import GradScaler, autocast`.
28. Instantiate the gradient scaler before the epoch loop: `scaler = GradScaler()`.
29. Wrap only the forward pass in the `autocast` context manager: `with autocast(): outputs = model(...); loss = outputs.loss`.
30. Replace the plain `loss.backward()` call with `scaler.scale(loss).backward()`.
31. Replace `optimizer.step()` with `scaler.step(optimizer)` followed by `scaler.update()`.
32. Run the first epoch with AMP enabled.
33. Record peak GPU memory after the first AMP epoch: `print(torch.cuda.max_memory_allocated(0) / 1e6)` and compare to the non-AMP run noted in step 25.
34. Observe in the timing cells (pre-written) whether step wall-clock time is faster with AMP than without.
35. Note in the narrative cell why AMP reduces memory: float16 activations use half the memory of float32, while float32 is preserved for weight updates via the scaler to maintain numerical stability.

**Section 4 — Running Training and Monitoring GPU (15 min)**

36. Open a terminal tab alongside the notebook: click **File > New > Terminal** in the Jupyter menu bar.
37. In the terminal, start the real-time GPU monitor: `watch -n 2 nvidia-smi`.
38. Observe the initial GPU utilization percentage and memory usage before training begins.
39. Switch back to the notebook tab and start the full AMP training run (all three epochs).
40. Switch back to the terminal tab and observe GPU utilization rising above 80% during forward and backward passes.
41. Note that GPU utilization dips briefly between batches — this is the data loading and CPU-to-GPU transfer gap.
42. After training completes, switch back to the notebook and record the final training loss from the last printed step.
43. Run the pre-written loss curve cell: `import matplotlib.pyplot as plt; plt.plot(loss_history); plt.xlabel('Step'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.show()`.
44. Observe the downward trend across steps confirming the model is learning from the fine-tuning data.
45. Run `print(f"Peak GPU memory: {torch.cuda.max_memory_allocated(0) / 1e6:.1f} MB")` and record the value.
46. Return to the terminal and stop the `watch` loop with `Ctrl+C`.

### Key Takeaways

- A complete PyTorch fine-tuning loop requires six elements in sequence: DataLoader, zero_grad, forward pass, loss extraction, backward pass, optimizer step
- Automatic mixed precision reduces GPU memory consumption and increases training throughput with three targeted code changes: `GradScaler`, `autocast` context, and scaled backward/step calls
- `nvidia-smi` provides real-time GPU utilization and memory data from outside the notebook — a critical monitoring tool during production AI workloads
- High GPU utilization (above 80%) during training confirms the GPU is not bottlenecked waiting for data; low utilization suggests a data pipeline or batch size problem
- Training loss should decrease across epochs; a flat or rising loss curve signals a learning rate, data, or optimizer configuration issue

### Infrastructure Notes

- The pre-tokenized dataset at `/data/tokenized_dataset.pt` was prepared and staged by Ansible automation before students arrived; the notebook loads it directly with `torch.load()` — students do not run tokenization in this lab
- `nvidia-smi` is available in the Jupyter terminal because the workbench pod is scheduled on a GPU node with the NVIDIA device plugin; the terminal inherits GPU device access from the pod's resource limits
- Automatic mixed precision requires a GPU with CUDA compute capability 7.0 or higher (Volta architecture or newer); the GPU nodes in the lab environment meet this requirement
