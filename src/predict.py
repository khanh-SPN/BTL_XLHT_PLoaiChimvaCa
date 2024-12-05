import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

MODEL_PATH = "./models/best_model.h5"
model = load_model(MODEL_PATH)

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Tiền xử lý ảnh để đưa vào mô hình.
    Args:
        image_path (str): Đường dẫn ảnh
        target_size (tuple): Kích thước đầu vào của mô hình

    Returns:
        numpy array: Ảnh đã được chuẩn hóa
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
    return img_array

def predict(image_path):
    """
    Dự đoán ảnh thuộc lớp nào.
    Args:
        image_path (str): Đường dẫn ảnh

    Returns:
        str: Kết quả dự đoán
    """
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    class_label = "Bird" if prediction[0][0] > 0.5 else "Fish"
    return class_label
