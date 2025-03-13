import tensorflow as tf
import kagglehub
import matplotlib.pyplot as plt

# Download dataset
dataset_path = kagglehub.dataset_download("sanidhyak/human-face-emotions")
print("Dataset Path:", dataset_path)

def load_dataset(directory, image_size=(128, 128), batch_size=32, validation_split=0.2):
    """
    Load images from a directory and prepare them for training.
    """
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="training",
        seed=42  # Ensuring reproducibility
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="validation",
        seed=42
    )

    return train_dataset, val_dataset

# Load dataset
train_dataset, val_dataset = load_dataset(dataset_path)

# Get class names
class_names = train_dataset.class_names
num_classes = len(class_names)
print("Class Names:", class_names)

# Define an augmentation layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
    tf.keras.layers.experimental.preprocessing.RandomBrightness(0.2),
])

# Define the CNN model with augmentation
model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)),  # Normalize pixel values
    data_augmentation,  # Augmentation applied here
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Dropout to prevent overfitting
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on augmented dataset
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10  # You can increase epochs for better results
)

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('CNN Training Progress with Augmentation')
plt.show()

# Save the model
model.save("cnn_emotion_classifier_augmented.h5")
