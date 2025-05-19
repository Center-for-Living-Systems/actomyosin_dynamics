# Auto-Encoders for actomyosin data
Understanding actomyosin dynamics is fundamental to studying cell motility, morphogenesis, and mechanical force generation in living tissues. This project applies auto-encoder architectures to learn unsupervised latent representations of actomyosin-rich regions in time-lapse microscopy datasets. These representations are designed to capture spatial and temporal patterns, compress the data meaningfully, and potentially support downstream tasks like classification, clustering, or anomaly detection.

### Data
We use high-resolution fluorescence microscopy image sequences from the Center for Living Systems (CLS) at the University of Chicago, where actin and myosin dynamics are visualized in various biological contexts:
Embryonic tissues, including gastrulating zebrafish and gastruloids
Actomyosin cortical flows 

### Goal
Develop 2D and 3D auto-encoders to learn meaningful representations of actomyosin structures
Explore 3D auto-encoders to account for dynamic behavior
Visualize and interpret the latent space to uncover biological patterns
Enable unsupervised clustering or anomaly detection using learned embeddings \
<img src="https://github.com/user-attachments/assets/b271f113-3394-4df4-ab95-ad9da2f3ff1d" style="width:40%;"/>

### Requirements
Python ≥ 3.9 \
PyTorch ≥ 1.13 or TensorFlow ≥ 2.10 \
NumPy, scikit-image, matplotlib, seaborn \
h5py, tifffile, imageio \
pytorch-lightning or keras \
umap-learn or scikit-learn for dimensionality reduction and clustering

### Output
Trained auto-encoder models (standard AE, VAE, temporal AE) \
2D/3D latent embeddings  \ 
UMAP/t-SNE plots showing latent structure  \
Clustering results for segmentation-free categorization of actomyosin states \
Code notebooks for training, evaluation, and visualization  \
Model weights and example datasets for reproduction \
<img src="https://github.com/user-attachments/assets/877df658-53dc-47a6-99d4-30a26b39a611" style="width:60%;"/>  \
<img src="https://github.com/user-attachments/assets/8eb8f9af-f33d-415c-bbfd-bc455430c850" style="width:60%;"/>  

## References
[1] https://github.com/hwalsuklee/tensorflow-mnist-VAE  
[2] https://www.slideshare.net/NaverEngineering/ss-96581209  
[3] https://www.tensorflow.org/alpha/tutorials/generative/cvae  
