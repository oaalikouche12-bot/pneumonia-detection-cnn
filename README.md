# Pneumonia Detection in Chest Radiographs using CNN

Deep learning project for **binary classification of chest radiographs** using a **Convolutional Neural Network (CNN)** and the **RSNA Pneumonia Detection Challenge** dataset.

## Objective

The goal of this project is to build a model capable of classifying chest X-ray images into two categories:

- **0 = No pneumonia**
- **1 = Pneumonia**

Although the original RSNA competition was designed as a **detection and localization** task, in this project the problem has been adapted to a simpler version more aligned with the academic content of **CNNs**, treating it as a **binary classification** problem.

## Dataset

This project uses the:

**RSNA Pneumonia Detection Challenge**

The dataset contains chest radiographs in **DICOM** format together with CSV files containing labels and detailed class information.

### Main files used

- `stage_2_train_images/`
- `stage_2_test_images/`
- `stage_2_train_labels.csv`
- `stage_2_detailed_class_info.csv`

## Methodology

The general workflow of the project is the following:

1. Load the dataset from Google Drive
2. Extract the ZIP file
3. Perform an initial exploration of the dataset structure
4. Read and process the labels
5. Convert the original problem into a **binary classification** task
6. Build a final table containing:
   - `patientId`
   - `Target`
   - `class`
   - `image_path`
7. Visualize DICOM radiographs
8. Preprocess the images:
   - DICOM reading
   - resizing to `128x128`
   - normalization
9. Create a balanced sample
10. Split the data into:
   - training
   - validation
   - test
11. Train a **basic CNN**
12. Evaluate the model using classification metrics

## Model Architecture

The main architecture used in this project is a **basic CNN**, following the core concepts studied in class:

- `Conv2D`
- `MaxPooling2D`
- `Conv2D`
- `MaxPooling2D`
- `Flatten`
- `Dense`
- `Dropout`
- `Dense(sigmoid)`

This architecture makes it possible to work with the fundamental concepts of convolutional neural networks:

- feature extraction
- spatial reduction through pooling
- flattening activation maps
- final classification using dense layers

## Results

The basic CNN achieved a reasonable performance on the test set, with balanced metrics across both classes.

### Main results

- **Test accuracy:** ~0.69
- **Precision:** ~0.68 - 0.69
- **Recall:** ~0.68 - 0.69
- **F1-score:** ~0.68 - 0.69

The confusion matrix showed a fairly balanced behavior between false positives and false negatives, suggesting that the model was not strongly biased toward only one class.

## Interpretation

The results indicate that a simple CNN is able to learn relevant visual patterns from chest radiographs and distinguish, to some extent, between pneumonia and non-pneumonia cases.

However, signs of **overfitting** were also observed:

- training loss decreased continuously,
- validation loss worsened after the first few epochs,
- validation accuracy stopped improving.

This suggests that the model learned the training data reasonably well, but had difficulty generalizing to unseen images.

## Second Model Tested

A more complex CNN architecture was also tested, including:

- one additional convolutional layer,
- `BatchNormalization`,
- and `GlobalAveragePooling2D`

However, this second version did not improve performance and collapsed into predicting a single class. For that reason, the **basic CNN** was considered the best baseline model in this project.

## Technologies Used

- Python
- Google Colab
- NumPy
- Pandas
- Matplotlib
- OpenCV
- pydicom
- scikit-learn
- TensorFlow / Keras

## Repository Structure

```text
.
├── deteccion_neumonia_radiografias_torax_cnn.ipynb
└── README.md
