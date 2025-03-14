# Image Augmentation with TensorFlow/Keras and # Emotion Classification Using CNN with Data Augmentation

## Overview
This project performs image augmentation on a set of images using TensorFlow/Keras. The script selects five random images from a specified dataset directory, applies various augmentation techniques, and displays the original and augmented images in a grid format.

Data Augmentation is is the process of artificially generating new data from existing data, primarily to train new machine learning (ML) models. ML models require large and varied datasets for initial training, but sourcing sufficiently diverse real-world datasets can be challenging because of data silos, regulations, and other limitations. Data augmentation artificially increases the dataset by making small changes to the original data( Definition by AWS).
Installation Guide

##Create a Virtual Environment
It is recommended to create a virtual environment to keep dependencies isolated.

For Windows (Command Prompt or PowerShell):

python -m venv emotion_env
emotion_env\Scripts\activate
For macOS/Linux (Terminal):

python3 -m venv emotion_env
source emotion_env/bin/activate

## Install Dependencies
Run the following command to install all required packages:

pip install opencv-python numpy matplotlib tensorflow scikit-learn
Dataset

Dataset Structure
The dataset consists of three folders inside a master folder (Dataset/), each containing images of human faces expressing the mentioned emotion:

Dataset/
    angry/
        image1.jpg
        image2.jpg
        ...
    happy/
        image1.jpg
        image2.jpg
        ...
    sad/
        image1.jpg
        image2.jpg
        ...
Dataset Details
Version: 1.0
Size: 85.67 MB
Last Update: 29/06/2023
Download Link: https://www.kaggle.com/datasets/sanidhyak/human-face-emotions


## Dependencies
Ensure you have the following Python packages installed before running the script:

- `os` (built-in)
- `random` (built-in)
- `cv2` (OpenCV)
- `numpy`
- `matplotlib`
- `tensorflow`

Install missing dependencies using:
```bash
pip install opencv-python numpy matplotlib tensorflow
```

## Features
- Loads images from the specified dataset directory.
- Randomly selects five images for augmentation.
- Applies the following augmentation techniques:
  - Rotation (up to 30 degrees)
  - Horizontal flipping
  - Brightness adjustment
  - Zoom (up to 20%)
  - Addition of Gaussian noise
- Displays the original and augmented images in a structured layout.

## Script Breakdown
### 1. Define Dataset Path
```python
DATASET_PATH = "Dataset"
```
This variable specifies the directory where the images are stored.

### 2. Load and List Images
The script iterates through the dataset directory and collects all image file paths with extensions `.jpg`, `.png`, or `.jpeg`.
```python
all_images = []
for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            all_images.append(os.path.join(root, file))
```
### 3. Random Image Selection
```python
random_images = random.sample(all_images, 5)
```
Five images are randomly selected from the dataset for augmentation.

### 4. Load Images
```python
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (128, 128))  # Resize for uniformity
    return img
```
This function reads an image using OpenCV, converts it to RGB format, and resizes it to 128x128 pixels.

### 5. Define Augmentation Transformations
Augmentations are performed using the `ImageDataGenerator` class from TensorFlow/Keras:
```python
data_gen = ImageDataGenerator(
    rotation_range=30,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    zoom_range=0.2
)
```

### 6. Add Gaussian Noise
A custom function adds Gaussian noise to an image:
```python
def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.uint8)
    noisy_image = cv2.add(image, gauss)
    return np.clip(noisy_image, 0, 255)
```

### 7. Apply Augmentations and Display Results
Each selected image undergoes different augmentation techniques. The original and augmented images are displayed using Matplotlib:
```python
fig, axes = plt.subplots(5, 6, figsize=(15, 12))
axes = axes.flatten()

for idx, image_path in enumerate(random_images):
    original_img = load_image(image_path)
    augmented_images = [
        data_gen.random_transform(original_img),  # Rotation
        cv2.flip(original_img, 1),  # Horizontal Flip
        data_gen.random_transform(original_img),  # Brightness Adjustment
        data_gen.random_transform(original_img),  # Zoom
        add_gaussian_noise(original_img)  # Gaussian Noise
    ]

    axes[idx * 6].imshow(original_img)
    axes[idx * 6].set_title("Original")
    titles = ["Rotated", "Flipped", "Brightness", "Zoom", "Noise"]

    for j, aug_img in enumerate(augmented_images):
        axes[idx * 6 + j + 1].imshow(aug_img.astype(np.uint8))
        axes[idx * 6 + j + 1].set_title(titles[j])

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
```

## Usage
1. Place your dataset of images in the `Dataset/` directory.
2. Run the script:
```bash
python script.py
```
3. The script will display the original images along with their augmented versions.

## Notes
- Ensure `Dataset/` contains sufficient images to allow random selection.
- You can modify augmentation parameters by adjusting `ImageDataGenerator` settings.



# Emotion Classification Using CNN with Data Augmentation

## Overview
This project implements a Convolutional Neural Network (CNN) for classifying emotions (angry, happy, sad) using image data. The model is trained on both the original dataset and an augmented dataset, with results compared to analyze the impact of augmentation.

## Dependencies
Ensure you have the following libraries installed before running the script:
```bash
pip install numpy opencv-python tensorflow matplotlib scikit-learn
```

## Dataset Structure
The dataset should be organized as follows:
```
Dataset/
    angry/
        image1.jpg
        image2.jpg
        ...
    happy/
        image1.jpg
        image2.jpg
        ...
    sad/
        image1.jpg
        image2.jpg
        ...
```
Each subfolder represents a class label and contains images related to that emotion.

## Features
- Loads images from the dataset and assigns labels based on folder names.
- Preprocesses images (resizing, normalization, and one-hot encoding).
- Splits the data into training and validation sets.
- Defines and trains a CNN model.
- Applies data augmentation techniques such as:
  - Rotation (up to 30 degrees)
  - Width and height shifting
  - Horizontal flipping
  - Brightness adjustment
  - Zooming
- Trains the same CNN model on both original and augmented datasets.
- Compares accuracy and loss performance between both models.

## Code Breakdown
### 1. Load and Preprocess Images
- Images are loaded from `Dataset/` and resized to 48x48 pixels.
- Labels are assigned based on folder names and one-hot encoded.
- Data is split into training and validation sets (80-20 split).

```python
images, labels = [], []
for label_idx, label in enumerate(emotion_labels):
    folder_path = os.path.join(DATASET_PATH, label)
    for image_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, image_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (48, 48))
        images.append(img)
        labels.append(label_idx)
```

### 2. Define and Train CNN Model
- The model consists of three convolutional layers followed by max pooling.
- Fully connected layers with dropout are used to prevent overfitting.

```python
def build_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(emotion_labels), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```

### 3. Train CNN on Original Dataset
```python
cnn_original = build_cnn()
history_original = cnn_original.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=1
)
```

### 4. Apply Data Augmentation
- Augmented training images are generated using `ImageDataGenerator`.

```python
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    zoom_range=0.2
)
```

### 5. Train CNN on Augmented Dataset
```python
augmented_train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
cnn_augmented = build_cnn()
history_augmented = cnn_augmented.fit(
    augmented_train_generator,
    validation_data=(X_val, y_val),
    epochs=20,
    verbose=1
)
```

### 6. Compare Results
- Accuracy and loss curves are plotted for both models.
```python
def plot_training_results(history, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(history.history['accuracy'], label="Train Accuracy")
    ax[0].plot(history.history['val_accuracy'], label="Val Accuracy")
    ax[0].set_title(f"{title} - Accuracy")
    ax[0].legend()
    ax[1].plot(history.history['loss'], label="Train Loss")
    ax[1].plot(history.history['val_loss'], label="Val Loss")
    ax[1].set_title(f"{title} - Loss")
    ax[1].legend()
    plt.show()
```

- The final validation accuracy of both models is compared:
```python
final_acc_original = history_original.history['val_accuracy'][-1]
final_acc_augmented = history_augmented.history['val_accuracy'][-1]
print(f"Final Validation Accuracy (Original Data): {final_acc_original:.4f}")
print(f"Final Validation Accuracy (Augmented Data): {final_acc_augmented:.4f}")
```

### Results Interpretation
- If the augmented model performs better, augmentation improved generalization.
- If not, the augmentation parameters might need tuning.

```python
if final_acc_augmented > final_acc_original:
    print("Augmentation improved the model's performance!")
else:
    print("The model trained on original data performed better!")
```

## Usage
1. Place your dataset in `Dataset/` following the given structure.
2. Run the script:
```bash
python script.py
```
3. The script will train and evaluate two CNN models and compare their performance.

## Notes
- Adjust the number of epochs based on dataset size.
- Modify augmentation parameters if needed.
- Ensure you have sufficient training data for meaningful comparisons.

## License
This project is open-source and free to use and modify.



