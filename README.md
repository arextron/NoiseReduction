# Noise Reduction Using Neural Networks
Members of the team:
|  |  Full Name  |   
| :---: | :---: |  
| 1   | Aryan Awasthi |  
| 2   | Muqaddas Preet Singh |  
| 3   | Harsukhvir Singh Grewal |  

Selected Papers/Documents that you can start with:
- [Development of Neural Networks for Noise Reduction](https://ccis2k.org/iajit/PDF/vol.7,no.3/945.pdf)
- [A signal processing interpretation of noise-reduction convolutional neural networks](https://arxiv.org/pdf/2307.13425)
- [Transient Noise Reduction Using a Deep Recurrent Neural Network: Effects on Subjective Speech Intelligibility and Listening Comfort](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8642050/)
- [An analysis of a noise reduction neural network](https://ieeexplore.ieee.org/document/266851)
- [Deep neural network algorithms for noise reduction and their application to cochlear implants](https://www.biorxiv.org/content/10.1101/2022.08.25.504678v1.full)
- [A Noise Filtering Method Using Neural Networks](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2003-Zeng-SCTIMRA.pdf)
- [A noise robust convolutional neural network for image classification](https://www.sciencedirect.com/science/article/pii/S2590123021000268)
- [Low-complexity artificial noise suppression methods for deep learning-based speech enhancement algorithms](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-021-00204-9)

Added Papers:
- [Signal speech reconstruction and noise removal using convolutional denoising audioencoders with neural deep learning](https://www.researchgate.net/publication/331839796_Signal_Speech_Reconstruction_and_Noise_removal_using_Convolutional_Denoising_Audioencoders_with_Neural_Deep_Learning)
- [Noise removal from Audio using CNN and Denoiser](https://immohann.github.io/Portfolio/Denoiser.pdf)


# Trasient Noise Reduction with Neural Networks

This repository contains implementations of different deep learning models for transient noise reduction in audio signals. The models are applied to the [CSTR VCTK dataset](https://datashare.ed.ac.uk/handle/10283/3443) to preprocess and enhance audio quality. The `.wav` files in the dataset are stored in the `wav48` folder inside subdirectories named `p225` to `p271`.

## Models

The following models are implemented in this repository:

1. **Convolutional Neural Network (CNN)**:
   - File: `CNN_F.ipynb`
   - A CNN-based approach to denoise audio signals by learning features from spectrograms.

2. **Recurrent Neural Network (RNN)**:
   - File: `NoiseReductionRNN_F.ipynb`
   - A recurrent model designed to capture temporal dependencies in audio signals for noise reduction.

3. **Multi-Layer Perceptron (MLP)**:
   - File: `MLP_F.ipynb`
   - A dense neural network used for learning mappings between noisy and clean audio features.


## Dataset

The models utilize the [CSTR VCTK dataset]([https://datashare.ed.ac.uk/handle/10283/3443](https://www.kaggle.com/datasets/pratt3000/vctk-corpus?select=VCTK-Corpus)) for training and evaluation. The dataset contains high-quality speech data with different accents, stored in `.wav` files under the `wav48` folder.

## How to Use

1. **Prepare the Dataset**:
   - Download and extract the [CSTR VCTK dataset]([https://datashare.ed.ac.uk/handle/10283/3443](https://www.kaggle.com/datasets/pratt3000/vctk-corpus?select=VCTK-Corpus)).
   - Place the `wav48` folder in the root directory.
   - Place the noise Folder in the dataset.

2. **Run the Notebooks**:
   - Each notebook (`CNN_F.ipynb`, `NoiseReductionRNN_F.ipynb`, `MLP_F.ipynb`) is self-contained and can be executed independently to train and evaluate the respective models.

3. **Add Transient Noise**:
   - To train models for transient noise reduction, augment the dataset with synthetic transient noise.

4. **Evaluate Performance**:
   - Evaluate the models on clean and noisy audio files to analyze their performance.

## Requirements

Install the following dependencies before running the notebooks:

- Python 3.8+
- TensorFlow/PyTorch
- NumPy
- Librosa
- Matplotlib
- Jupyter Notebook

## Future Work

- Fine-tune models for real-world noisy environments.
- Compare model performance on different types of transient noise.
- Explore hybrid models combining CNN, RNN, and MLP architectures.

## Contact

For any inquiries or suggestions, feel free to reach out.


