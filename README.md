# Detecting-Parkinson's-Disease-from-EEG-Signals
AI model uses DWT/Entropy features from raw EEG signals to detect PD. The XGBoost classifier with Mutual Information feature selection achieved an exceptional 0.9905 F1-Score. The pipeline includes SHAP/LIME for explainability

Parkinsons Detection using Raw EEG Signals 🧠
This project implements a robust and interpretable machine learning pipeline for the precise classification of Parkinson's Disease (PD) patients and healthy controls based on their resting-state Electroencephalography (EEG) signals. This addresses the urgent demand for objective, non-invasive, and quantitative biomarkers for PD diagnosis, moving beyond traditional subjective clinical methods

💡 Core Contributions
Proposing an interpretable and reproducible diagnostic framework for PD detection.

Integrating Explainable Artificial Intelligence (XAI) techniques (SHAP and LIME) to provide transparent, clinically understandable insights into model decisions.
Employing a pipeline that incorporates a hybrid of Discrete Wavelet Transform (DWT) and a suite of entropy-based measures for rich signal dynamics and feature extraction.
Systematically comparing multiple classifiers and feature selection schemes to find the strongest biomarkers.

🛠 Methodology
The proposed framework follows a structured signal-processing and machine-learning pipeline.
1. Dataset DetailsDataset: OpenNeuro's Rest eyes open dataset (ds004584)888.Participants: 149 subjects: 100 individuals with PD and 49 healthy controls9.Recording: EEG was recorded with a 64-channel BrainVision cap10.
2. EEG PreprocessingThe pipeline uses MNE-Python for transparency and reproducibility.Filtering: 1–40 Hz band-pass and 50 Hz notch filtering.Denoising: Variance-based bad-channel interpolation and common average re-referencing were performed
3. Artifact Removal: ICA-based artifact removal was employed to minimize physiological artifacts like ocular or cardiac origins
4. Segmentation: Cleaned continuous data were divided into 5-second epochs.
5. Feature ExtractionEach artifact-free epoch is decomposed using a four-level Daubechies-4 (db4) Discrete Wavelet Transform (DWT).
6. The computed features measure both signal power and entropy-based irregularity:Energy/Power: Energy (E_i) and Log Band Power (P_i) capture amplitude-based differences
7. Entropy Measures: Eight complementary energy and entropy measures were computed, including Sure Entropy (SE), Shannon Entropy (ShEn), Log Energy Entropy (LEE), and Transformed Shannon Entropy (TShEn).
8. Feature Selection and Classification Feature Selection Methods Compared: Four complementary methods were applied: Mutual Information (MI), Recursive Feature Elimination (RFE), ANOVA F-test, and Chi-Square Test.
9. Classifiers Compared: Five classifiers were trained and evaluated: SVM (RBF Kernel), Logistic Regression, k-Nearest Neighbors (k-NN), Random Forest, and XGBoost

🏆 Key Results
The best performance was achieved by combining the XGBoost classifier with the Mutual Information feature selection method222222.
Test F1-Score: 0.9905
Test Accuracy: 0.9905 24
Explainability (XAI) Insights
The integration of SHAP and LIME provided crucial transparency.
Key Biomarker Identified: The most influential global feature was the Sure Entropy (SE) of the D_1 wavelet coefficient at the F8 electrode (F8_cD1_suen).SHAP Trend: Low values of this feature consistently produced positive SHAP values, strongly indicating a prediction of Parkinson's Disease.
