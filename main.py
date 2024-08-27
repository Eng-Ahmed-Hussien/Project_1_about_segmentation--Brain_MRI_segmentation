import os  # import for interacting with the operating system
import cv2  # import for computer vision tasks
import numpy as np  # import for numerical operations
import random  # import for generating random numbers
import matplotlib.pyplot as plt  # import for plotting
from skimage import feature, exposure  # import for image processing
from skimage.measure import regionprops, label  # import for measuring properties of labeled regions
from skimage.morphology import remove_small_objects, skeletonize  # import for morphological operations
from skimage.feature import local_binary_pattern, hog  # import for feature extraction
from sklearn.svm import SVC  # import for Support Vector Classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score  # import for model selection and evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # import for evaluating model performance
from sklearn.preprocessing import LabelEncoder, StandardScaler  # import for preprocessing
from sklearn.ensemble import RandomForestClassifier  # import for Random Forest Classification
from imblearn.over_sampling import SMOTE  # import for handling imbalanced datasets
from scipy.stats import skew, kurtosis  # import for statistical functions
from joblib import dump  # import for saving models
import warnings  # import for managing warnings
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

def load_images_from_folder(folder, target_size=(128, 128)):
    images = []  # list to hold images
    labels = []  # list to hold labels
    for label in os.listdir(folder):  # iterate through folders
        label_folder = os.path.join(folder, label)  # get the path to the label folder
        for filename in os.listdir(label_folder):  # iterate through files in the label folder
            img = cv2.imread(os.path.join(label_folder, filename), cv2.IMREAD_GRAYSCALE)  # read image in grayscale
            if img is not None:  # check if image is read successfully
                img_resized = cv2.resize(img, target_size)  # resize image to target size
                images.append(img_resized)  # append image to list
                labels.append(label)  # append label to list
    return images, labels  # return lists of images and labels

# Example: loading images from the dataset folder
dataset_path = "./brain_tumor/Training"  # path to the dataset
images, labels = load_images_from_folder(dataset_path)  # load images and labels

images = np.array(images)  # convert list of images to numpy array
labels = np.array(labels)  # convert list of labels to numpy array

print(f"Loaded {len(images)} images with corresponding labels.")  # print number of loaded images
print(f"Image shape: {images[0].shape}")  # print shape of the first image

plt.figure(figsize=(15, 5))  # create a figure for plotting
for i in range(5):  # iterate through 5 images
    somenum = random.randint(0, len(images) - 1)  # pick a random image
    plt.subplot(1, 5, i+1)  # create subplot
    plt.imshow(images[somenum], cmap='gray')  # display image in grayscale
    plt.title(f"Label: {labels[somenum]}")  # set title as the label
    plt.axis('off')  # hide axes

plt.show()  # display the plot

def apply_clahe(images):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # create CLAHE object
    return np.array([clahe.apply(image) for image in images])  # apply CLAHE to each image

images_clahe = apply_clahe(images)  # apply CLAHE to images

for i in range(5):  # iterate through 5 images
    plt.figure(figsize=(5, 5))  # create a figure for plotting
    plt.subplot(1, 2, 1)  # create subplot for original image
    plt.imshow(images[i], cmap='gray')  # display original image in grayscale
    plt.title('Original Image')  # set title
    plt.subplot(1, 2, 2)  # create subplot for CLAHE image
    plt.imshow(images_clahe[i], cmap='gray')  # display CLAHE enhanced image
    plt.title('CLAHE Enhanced Image')  # set title
    plt.show()  # display the plot

def enhanced_region_growing(image, threshold=0.3):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  # apply Gaussian blur
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # apply Otsu's thresholding
    otsu_threshold = _ / 255.0  # normalize threshold
    final_threshold = (threshold + otsu_threshold) / 2  # calculate final threshold
    labels = np.zeros_like(image)  # initialize labels
    seeds = np.where(image > final_threshold * image.max())  # find seeds based on threshold
    for seed in zip(*seeds):  # iterate through seeds
        if labels[seed] == 0:  # if seed is not labeled
            seed_label = label(image > final_threshold * image.max())  # label regions
            labels[seed_label == seed_label[seed]] = np.max(labels) + 1  # assign new label
    return labels  # return labeled image

segmented_images = [enhanced_region_growing(img) for img in images_clahe]  # segment images

for i in range(5):  # iterate through 5 images
    plt.figure(figsize=(5, 5))  # create a figure for plotting
    plt.subplot(1, 2, 1)  # create subplot for original image
    plt.imshow(images_clahe[i], cmap='gray')  # display original image in grayscale
    plt.title('Original Image')  # set title
    plt.subplot(1, 2, 2)  # create subplot for segmented image
    plt.imshow(segmented_images[i], cmap='gray')  # display segmented image
    plt.title('Segmented Image')  # set title
    plt.show()  # display the plot

def enhanced_lbp(image, P=24, R=3):
    lbp = local_binary_pattern(image, P, R, method='uniform')  # calculate LBP
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))  # calculate histogram
    hist = hist.astype('float')  # convert to float
    hist /= (hist.sum() + 1e-6)  # normalize histogram
    return hist  # return histogram

def visualize_lbp(lbp_histogram, image, P=24, R=3):
    lbp = local_binary_pattern(image, P, R, method='uniform')  # calculate LBP
    plt.figure(figsize=(12, 6))  # create a figure for plotting
    plt.subplot(1, 3, 1)  # create subplot for original image
    plt.imshow(image, cmap='gray')  # display original image in grayscale
    plt.title('Original Image')  # set title
    plt.subplot(1, 3, 2)  # create subplot for LBP image
    plt.imshow(lbp, cmap='nipy_spectral')  # display LBP image
    plt.title('LBP Image')  # set title
    plt.subplot(1, 3, 3)  # create subplot for LBP histogram
    plt.bar(range(len(lbp_histogram)), lbp_histogram)  # plot histogram
    plt.title(f'LBP Histogram (P={P})')  # set title
    plt.xlabel('LBP Value')  # set x-axis label
    plt.ylabel('Frequency')  # set y-axis label
    plt.show()  # display the plot

def validate_lbp_features(images):
    for idx, img in enumerate(images[:5]):  # iterate through 5 images
        lbp_histogram = enhanced_lbp(img)  # calculate LBP histogram
        print(f"Validating Image {idx + 1}")  # print validation info
        visualize_lbp(lbp_histogram, img, P=24, R=3)  # visualize LBP features

validate_lbp_features(images_clahe)  # validate LBP features

def enhanced_morphological_features(segmented_img, intensity_image):
    features = []  # list to hold features
    labeled_img = label(segmented_img)  # label segmented image
    skeleton = skeletonize(segmented_img)  # skeletonize segmented image
    regions = regionprops(labeled_img, intensity_image=intensity_image)  # get region properties

    for region in regions:  # iterate through regions
        if region.area == 0 or region.perimeter == 0:  # skip if area or perimeter is zero
            continue

        solidity = region.solidity  # get solidity
        extent = region.extent  # get extent
        euler_number = region.euler_number  # get Euler number

        glcm = feature.graycomatrix(intensity_image[region.bbox[0]:region.bbox[2],
                                                   region.bbox[1]:region.bbox[3]],
                                    [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                                    symmetric=True, normed=True)  # calculate GLCM
        contrast = feature.graycoprops(glcm, 'contrast').mean()  # get contrast
        dissimilarity = feature.graycoprops(glcm, 'dissimilarity').mean()  # get dissimilarity
        homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()  # get homogeneity
        energy = feature.graycoprops(glcm, 'energy').mean()  # get energy
        correlation = feature.graycoprops(glcm, 'correlation').mean()  # get correlation

        intensities = intensity_image[region.coords[:, 0], region.coords[:, 1]]  # get pixel intensities
        intensity_mean = np.mean(intensities)  # calculate mean intensity
        intensity_std = np.std(intensities)  # calculate standard deviation of intensity

        # Handle precision loss
        try:
            if len(np.unique(intensities)) < 2:  # Check for nearly identical values
                intensity_skewness = 0  # set skewness to 0
                intensity_kurtosis = 0  # set kurtosis to 0
            else:
                intensity_skewness = skew(intensities)  # calculate skewness
                intensity_kurtosis = kurtosis(intensities)  # calculate kurtosis
        except Exception as e:  # handle exceptions
            print(f"Error calculating skewness/kurtosis: {e}")  # print error message
            intensity_skewness = 0  # set skewness to 0
            intensity_kurtosis = 0  # set kurtosis to 0

        feature_dict = {  # create dictionary of features
            'Area': region.area,
            'Perimeter': region.perimeter,
            'Circularity': (4 * np.pi * region.area) / (region.perimeter ** 2 + 1e-10),
            'Eccentricity': region.eccentricity,
            'Major_Axis_Length': region.major_axis_length,
            'Minor_Axis_Length': region.minor_axis_length,
            'Solidity': solidity,
            'Extent': extent,
            'Euler_Number': euler_number,
            'Contrast': contrast,
            'Dissimilarity': dissimilarity,
            'Homogeneity': homogeneity,
            'Energy': energy,
            'Correlation': correlation,
            'Intensity_Mean': intensity_mean,
            'Intensity_Std': intensity_std,
            'Intensity_Skewness': intensity_skewness,
            'Intensity_Kurtosis': intensity_kurtosis,
        }

        features.append(feature_dict)  # append features to list

    return features  # return list of features

def calculate_hog(image):
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True)  # calculate HOG features
    return fd  # return HOG features

def gabor_features(image):
    num_scales = 4  # number of scales for Gabor filter
    num_orientations = 6  # number of orientations for Gabor filter
    gabor_responses = []  # list to hold Gabor responses

    for scale in range(num_scales):  # iterate through scales
        for orientation in range(num_orientations):  # iterate through orientations
            frequency = 0.1 / (2 ** scale)  # calculate frequency
            theta = orientation * np.pi / num_orientations  # calculate orientation angle
            kernel = cv2.getGaborKernel((21, 21), 5, theta, frequency, 0.5, 0, ktype=cv2.CV_32F)  # create Gabor kernel
            filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)  # apply Gabor filter
            gabor_responses.extend([filtered.mean(), filtered.std()])  # append mean and std to responses

    return np.array(gabor_responses)  # return Gabor responses

def enhanced_extract_features(images, segmented_images):
    all_features = []  # list to hold all features
    feature_names = []  # list to hold feature names

    for img, segmented_img in zip(images, segmented_images):  # iterate through images and segmented images
        lbp_hist = enhanced_lbp(img)  # calculate LBP histogram
        hog_features = calculate_hog(img)  # calculate HOG features
        gabor_feats = gabor_features(img)  # calculate Gabor features
        morph_features = enhanced_morphological_features(segmented_img, img)  # calculate morphological features

        if not morph_features:  # skip if no morphological features
            continue

        combined_features = np.concatenate([  # concatenate all features
            lbp_hist,
            hog_features,
            gabor_feats,
            list(morph_features[0].values())
        ])

        all_features.append(combined_features)  # append combined features to list

        if not feature_names:  # if feature names list is empty
            feature_names = (  # create feature names list
                    [f'LBP_{i}' for i in range(len(lbp_hist))] +
                    [f'HOG_{i}' for i in range(len(hog_features))] +
                    [f'Gabor_{i}' for i in range(len(gabor_feats))] +
                    list(morph_features[0].keys())
            )

    return np.array(all_features), feature_names  # return features and feature names

if __name__ == "__main__":  # execute script
    features, feature_names = enhanced_extract_features(images_clahe, segmented_images)  # extract features

    le = LabelEncoder()  # create label encoder
    encoded_labels = le.fit_transform(labels)  # encode labels
    X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)  # split data

    scaler = StandardScaler()  # create scaler
    X_train_scaled = scaler.fit_transform(X_train)  # scale training data
    X_test_scaled = scaler.transform(X_test)  # scale test data

    smote = SMOTE(random_state=42)  # create SMOTE object
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)  # resample training data

    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)  # create SVM classifier
    svm.fit(X_train_resampled, y_train_resampled)  # train SVM model
    svm_pred = svm.predict(X_test_scaled)  # predict with SVM
    print("SVM Accuracy:", accuracy_score(y_test, svm_pred))  # print SVM accuracy
    print(classification_report(y_test, svm_pred))  # print classification report

    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)  # create Random Forest classifier
    rf.fit(X_train_resampled, y_train_resampled)  # train Random Forest model
    rf_pred = rf.predict(X_test_scaled)  # predict with Random Forest
    print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))  # print Random Forest accuracy
    print(classification_report(y_test, rf_pred))  # print classification report

    feature_importance = rf.feature_importances_  # get feature importance
    sorted_idx = np.argsort(feature_importance)  # get sorted indices of feature importance
    pos = np.arange(sorted_idx.shape[0]) + .5  # positions for bars

    plt.figure(figsize=(12, 6))  # create a figure for plotting
    plt.barh(pos, feature_importance[sorted_idx], align='center')  # create horizontal bar plot
    plt.yticks(pos, np.array(feature_names)[sorted_idx])  # set y-axis ticks
    plt.xlabel('Feature Importance')  # set x-axis label
    plt.title('Feature Importance (Random Forest)')  # set title
    plt.tight_layout()  # adjust layout
    plt.show()  # display the plot

    dump(rf, 'best_rf_model.joblib')  # save the model
    print("Best model saved as 'best_rf_model.joblib'")  # print confirmation

    print("Script execution completed.")  # print completion message
