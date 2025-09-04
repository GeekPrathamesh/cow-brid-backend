import tensorflow as tf

from tensorflow import keras

# Fix imports
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
layers = keras.layers
models = keras.models


# Path to dataset
data_path = "data/"

# Data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_path,
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    data_path,
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("🚀 Training started...")
model.fit(train_data, validation_data=val_data, epochs=5)

# Save model
model.save("cow_model.h5")
print("✅ Model saved as cow_model.h5")
