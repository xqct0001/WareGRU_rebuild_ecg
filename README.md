# WareGRU_rebuild_ecg
This project is an open-source content for the reconstruction of electrocardiogram articles.

This project offers (WaveGRU-Net) models

# If it is helpful, please cite the following article

@article{XU2025110108,
title = {WaveGRU-Net: Robust non-contact ECG reconstruction via MIMO millimeter-wave radar and multi-scale semantic analysis},
journal = {Signal Processing},
volume = {237},
pages = {110108},
year = {2025},
issn = {0165-1684},
doi = {https://doi.org/10.1016/j.sigpro.2025.110108},
url = {https://www.sciencedirect.com/science/article/pii/S0165168425002221},
author = {Dan Xu and Yiming Xu and Kaijie Xu and Ze Hu and Mengdao Xing and Fulvio Gini and Maria Sabrina Greco},
keywords = {Non-contact vital sign monitoring, Non-contact ECG reconstruction, MIMO millimeter-wave radar, WaveGRU-Net},
abstract = {With the rising demand for telemedicine, non-contact heart beating monitoring has attracted significant interest due to its non-invasive and patient-friendly attributes. However, conventional approaches are typically limited to detecting the peaks of the Electrocardiogram (ECG), making the accurate extraction of ECG intervals challenging. This paper proposed a novel method for non-contact ECG signal reconstruction utilizing multiple-input-multiple-output millimeter-wave radar, enabling precise reconstruction of comprehensive ECG features and capturing nuanced variations in cardiac activity. First, Two-Dimensional beamforming is employed to enhance the radar signal of interest. The echo inevitably contains interference from random body movements and chest displacements caused by respiration. The interference from random body movements can be effectively suppressed by using a cumulative energy spectrum analysis. Next, the phase information representing the combined respiratory and cardiac micro-movements is extracted. Then, the phase is inputted into the WaveGRU-Net model, which is an advanced neural network based on the Convolutional Neural Network-Long Short-Term Memory architecture, to reconstruct heartbeat signals and ECG waveforms. The proposed method successfully separates respiratory and cardiac signals in the time-frequency domain, yielding a refined ECG reconstruction enriched with detailed semantic features that encapsulate subtle cardiac dynamics. Experimental results demonstrate the proposed method has strong semantic representation capabilities.}
}
