# DDPMSubtraction

This repository contains an implementation of a **conditional Denoising Diffusion Probabilistic Model (DDPM)** trained to generate the result of a subtraction operation between two handwritten digits.

Instead of generating an unconstrained MNIST digit, the model is conditioned on **two input digit images** and learns to generate an image corresponding to their subtraction result. Conditioning is implemented by concatenating the noisy target image with the two input images and injecting timestep information via sinusoidal embeddings and FiLM modulation.

The model was trained for **30 epochs on 100,000 samples** using an **NVIDIA A100 GPU** on a Lambda Cloud VM. After training, the model generates **highly realistic digit images**, with partial but non-deterministic adherence to the subtraction conditioning.

While the conditioning signal is not always strong enough to enforce the correct subtraction result, the model consistently produces structured, plausible digits and occasionally solves the subtraction task correctly. This behavior highlights known limitations of early conditioning strategies in diffusion models and motivates architectural extensions such as multi-scale conditioning or classifier-free guidance.
 

## Key Takeaways

1. A relatively small DDPM trained on 100,000 samples can generate highly realistic handwritten digits.
2. Conditioning via input concatenation can guide generation toward task-relevant outputs, but does not strictly enforce correctness.
3. Training diffusion models of this scale is computationally efficient on modern GPUs (A100).
4. Periodic sampling, checkpointing, and logging are essential for diagnosing training dynamics and conditioning behavior.


Note:
This project focuses on understanding the behavior and limitations of conditional diffusion models rather than optimizing task accuracy.
