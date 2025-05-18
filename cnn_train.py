import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# --- Klasör ve model ismi ayarları ---
model_save_dir = 'saved_models'
os.makedirs(model_save_dir, exist_ok=True)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_name = f"model_{current_time}"
model_path = os.path.join(model_save_dir, f"{model_name}.h5")

# --- Veri Artırma (Train) ve Normalizasyon (Valid/Test) ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    shear_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# --- Dizinler (kendine göre değiştir) ---
train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'

# --- Veri generatorları ---
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_generator = valid_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# --- Class weight hesapla ---
counter = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(counter), y=counter)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# --- Model mimarisi ---
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3),
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.3),

    layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(train_generator.num_classes, activation='softmax')
])

optimizer = Adam(learning_rate=0.0001)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Callbacks ---
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, min_delta=0.001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6, verbose=1)
model_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

callbacks = [early_stop, reduce_lr, model_checkpoint]

# --- Model Eğitimi ---
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=valid_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# --- Eğitim ve Doğrulama Performansı Grafik ---
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(model_save_dir, f"{model_name}_training_history.png"))
plt.show()

# --- Test Setinde Değerlendirme ---
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# --- Sonuçları Kaydet ---
with open(os.path.join(model_save_dir, f"{model_name}_results.txt"), "w") as f:
    f.write(f"Model: {model_name}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print(f"Model başarıyla kaydedildi: {model_path}")
print(f"Eğitim grafikleri ve sonuçlar '{model_save_dir}' klasörüne kaydedildi.")
