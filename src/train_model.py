import tensorflow as tf
from tensorflow.keras import layers, models
from data_preprocessing import create_data_generators

# Đường dẫn dữ liệu
DATA_DIR = "../data/train"
MODEL_SAVE_PATH = "../models/best_model.h5"

# Tạo các data generator
train_gen, val_gen = create_data_generators(DATA_DIR)

# Xây dựng mô hình
def build_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Phân loại nhị phân
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Huấn luyện mô hình
model = build_model()
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# Lưu mô hình
model.save(MODEL_SAVE_PATH)
print("Mô hình đã được lưu tại:", MODEL_SAVE_PATH)
