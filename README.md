# FRESH-BCI
Few-Shot Relational Inference Enables Minimal-Calibration EEG-to-Robot Control

This repository implements the FRESH model for EEG signal analysis with domain adversarial training, few-shot metric learning, and prototype refinement via message passing. The model integrates an EEG encoder with spectral and spatial attention, a domain classifier (using a gradient reversal layer), a relation network for computing pairwise similarity, and a message passing module to refine class prototypes.

## Overview

The key components of the model include:

- **Gradient Reversal Layer (GRL):**  
  A custom autograd function (`GradientReversalFunction`) that reverses gradients during backpropagation. It is used for domain adversarial training.

- **EEG Encoder (`EEGEncoder`):**  
  A CNN-based encoder that processes EEG trials (expected shape: `(batch, 1, T, 5, 5)`). It applies:
  - A temporal convolution (with kernel size 65 along time) to extract features.
  - A spatial convolution (5x5 kernel) to capture spatial information.
  - **Spectral Attention:** Applies attention weights across frequency bands.
  - **Spatial Attention:** Generates a spatial attention map over the electrode grid.
  - A final global average pooling and fully-connected layer to produce the embedding.

- **Domain Classifier (`DomainClassifier`):**  
  A simple fully-connected network that predicts domain labels (e.g., different sessions or subjects) from the embeddings. The input embedding is passed through a gradient reversal layer before classification.

- **Relation Network (`RelationNetwork`):**  
  Computes a relation (similarity) score between two embeddings, which is used for few-shot metric learning.

- **Message Passing Module (`MessagePassing`):**  
  Refines class prototypes by propagating messages among prototype embeddings based on their similarity to a query embedding.

- **Loss Functions:**  
  The repository also provides implementations for:
  - **Metric Loss:** Mean squared error loss computed over pairs of support embeddings.
  - **Sparsity Loss:** An L1 regularization loss on embeddings.
  - **Regularization Loss:** A consistency loss ensuring that relation scores remain stable under slight input perturbations.

- **FRESH Model (`FRESHModel`):**  
  Combines the above components into one complete model.

- **Tools and Source Code References:**  
  - sLORETA: Source localization for EEG signal analysis. Learn more at sLORETA.
  - BBCI Toolbox: For real-time EEG processing and BCI applications. Available at BBCI Toolbox.
