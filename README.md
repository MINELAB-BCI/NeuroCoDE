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


## Tools and Source Code References
- sLORETA(Standardized Low-Resolution Brain Electromagnetic Tomography): Source localization for EEG signal analysis. Learn more at [sLORETA](https://www.uzh.ch/keyinst/loreta).
 - sLORETA is a neuroimaging technique that estimates the three-dimensional distribution of electrical activity in the brain based on EEG measurements. It addresses the inverse problem by inferring the sources of the scalp-recorded electrical signals and standardizes the results to reduce localization bias. Although it provides low spatial resolution, sLORETA delivers a stable and reliable map of brain activity, making it widely used in both clinical research and neuroscience.
 - In this research, sLORETA was employed to assess the quality of the recorded brain signals and to verify that motor imagery was indeed occurring in the motor cortex. The validation results clearly demonstrate that the observed EEG patterns accurately reflect the neural activity associated with the cognitive processes underlying motor imagery, effectively supporting the feasibility of EEG-based robotic arm control using only the channels corresponding to the motor cortex.

- BBCI Toolbox: For real-time EEG processing and BCI applications. Available at [BBCI Toolbox](https://github.com/bbci/bbci_public).
