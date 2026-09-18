# GPU-Accelerated LLM Fine-Tuning with PyTorch

## Overview

This lab teaches intermediate Tech Sales professionals how to fine-tune large language models using PyTorch on GPU-accelerated OpenShift infrastructure managed by Red Hat OpenShift AI. It exists to give technical sellers a hands-on reference point for customer conversations about AI workloads on OpenShift.

Participants will verify GPU access in a dedicated OpenShift cluster, load pre-trained BERT and GPT-2 models from a pre-staged Hugging Face cache, run a fine-tuning training loop with automatic mixed precision, monitor GPU utilization in real time, and diagnose CUDA out-of-memory errors — completing the full fine-tuning lifecycle in approximately two hours.

## Target Audience

- **Role:** Tech Sales — solution architects, technical account managers, field engineers
- **Experience level:** Intermediate
- **What they already know:** Basic Python, general familiarity with ML concepts (what a model is, what training means), general OpenShift awareness
- **What they don't know:** GPU workload management on OpenShift, PyTorch fine-tuning workflows, CUDA memory optimization, RHOAI GPU scheduling

## Prerequisites

- Basic Python programming (read and modify scripts without hand-holding)
- Familiarity with machine learning concepts at a conceptual level
- No GPU expertise or deep OpenShift experience required

Can the lab validate these automatically? No — trust-based. Students self-assess readiness from the prerequisites page.

## Learning Objectives

1. **Implement** LLM fine-tuning workflows with PyTorch on GPU-accelerated OpenShift
2. **Implement** GPU memory optimization techniques for transformer models
3. **Implement** efficient training loops with automatic mixed precision
4. **Monitor** and debug GPU utilization during LLM training
5. **Analyze** and troubleshoot CUDA out-of-memory errors

## Content Type

Lab (hands-on)

## Products & Technologies

- Red Hat OpenShift Container Platform 4.21
- Red Hat OpenShift AI (RHOAI)
- PyTorch (upstream)
- NVIDIA CUDA (upstream)
- Hugging Face Transformers — BERT, GPT-2 (upstream)

## Module Map

| Module | Title | Duration |
|--------|-------|----------|
| 1 | PyTorch GPU Setup | 25 min |
| 2 | Loading Pre-trained Models | 30 min |
| 3 | Fine-tuning Workflow | 45 min |
| 4 | Optimization and Results | 20 min |
| — | **Total hands-on** | **~2 hours** |
| — | Intro / orientation | ~10 min |
| — | **Total lab** | **~2 hours 10 min** |

Module 1 covers GPU environment verification and basic PyTorch tensor operations on GPU. Module 2 loads BERT and GPT-2 from the pre-staged Hugging Face cache into GPU memory. Module 3 runs the core fine-tuning loop with automatic mixed precision (AMP) and live GPU utilization monitoring. Module 4 introduces memory profiling, gradient accumulation, and guided CUDA OOM error resolution.

Dataset preparation is handled by Ansible automation before students arrive — students start directly from model loading in Module 2.

## Difficulty Level

Intermediate

## Environment

**Learner view:** Each student is provisioned a dedicated OpenShift cluster with at least one GPU-enabled worker node managed by Red Hat OpenShift AI. A Jupyter notebook environment is pre-deployed with PyTorch, CUDA drivers, and all Python dependencies pre-installed. Pre-trained BERT and GPT-2 model weights are pre-downloaded and cached locally — no internet access to Hugging Face is required during the lab. A tokenized training dataset is pre-staged in a shared volume.

**Automation needed:** Yes
- Provision OCP cluster with GPU-enabled worker nodes
- Deploy RHOAI operator and configure GPU scheduling
- Pre-install PyTorch, CUDA, and notebook dependencies
- Pre-download and cache Hugging Face model weights
- Pre-tokenize and stage the training dataset

## Infrastructure Requirements

- **Cloud provider:** TBD — confirmed in infrastructure phase
- **Cluster type:** TBD — confirmed in infrastructure phase
- **OCP version:** TBD — confirmed in infrastructure phase
- **Topology:** TBD — confirmed in infrastructure phase
- **Sizing:** TBD — confirmed in infrastructure phase
- **Automation approach:** TBD — confirmed in infrastructure phase
- **AI/MaaS:** TBD — confirmed in infrastructure phase
- **External services:** TBD — confirmed in infrastructure phase
- **AAP version:** TBD — confirmed in infrastructure phase
- **Non-GA products:** TBD — confirmed in infrastructure phase
