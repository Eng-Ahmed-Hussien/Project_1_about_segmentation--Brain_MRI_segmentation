# Brain MRI Segmentation and Feature Extraction

## Project Overview

This project involves the segmentation and feature extraction of brain MRI images. It utilizes various image processing techniques such as CLAHE (Contrast Limited Adaptive Histogram Equalization), LBP (Local Binary Patterns), and morphological operations. The extracted features are then used to train classifiers like SVM (Support Vector Machine) and Random Forest to predict outcomes.

## Requirements

To run this project, you need to have the following tools and libraries installed:

- Python (version 3.x)
- Jupyter Notebook
- OpenCV
- NumPy
- Scikit-image
- Scikit-learn
- Matplotlib
- Imbalanced-learn
- Joblib

Install the required libraries using the following command:

```bash
pip install opencv-python numpy scikit-image scikit-learn matplotlib imbalanced-learn joblib
```

## Installation and Setup

Follow the steps below to set up and run the project:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Eng-Ahmed-Hussien/Project_1_about_segmentation--Brain_MRI_segmentation
   cd Project_1_about_segmentation--Brain_MRI_segmentation
   ```

2. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
   ```

3. **Install the Required Libraries**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook main.ipynb
   ```

## Steps to Follow

1. **Load and Preprocess Images**:

   - The notebook defines a function to load brain MRI images from a folder, resize them, and store them along with their labels.

2. **Visualize Sample Images**:

   - Randomly select and display sample images with their labels.

3. **Apply CLAHE**:

   - Enhance the contrast of images using CLAHE.

4. **Segment Images**:

   - Segment the images using an enhanced region-growing algorithm.

5. **Extract LBP and Morphological Features**:

   - Extract Local Binary Patterns (LBP) and morphological features from the images.

6. **Train Classifiers**:

   - Use the extracted features to train an SVM and Random Forest classifiers.

7. **Evaluate Models**:
   - Evaluate the accuracy of the classifiers and visualize feature importance.

## Visualization of the Output

The notebook provides visual outputs for the following:

- **Sample Images**: Displaying sample brain MRI images with labels.
- **Segmentation Results**: Visualizations showing segmented brain regions.
- **Feature Importance**: Charts showing the importance of features as determined by the Random Forest classifier.
- **Classification Results**: Confusion matrices and classification reports for the SVM and Random Forest models.

## Conclusion

This notebook provides a comprehensive guide to performing segmentation and feature extraction on brain MRI images, followed by training and evaluating classifiers. By following the steps outlined above, you will gain insights into the effectiveness of different image processing techniques and classifiers in medical image analysis.

## References

- **[Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)**
- **[Preprocessing](https://github.com/masoudnick/Brain-Tumor-MRI-Classification/blob/main/Preprocessing.py)**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
