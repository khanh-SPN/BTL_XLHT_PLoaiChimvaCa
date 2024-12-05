import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(data_dir, target_size=(224, 224), batch_size=32):
    """
    Tạo các generator cho dữ liệu huấn luyện, xác thực và kiểm tra.
    Args:
        data_dir (str): Đường dẫn tới thư mục chứa dữ liệu
        target_size (tuple): Kích thước hình ảnh đầu vào
        batch_size (int): Số lượng mẫu mỗi batch

    Returns:
        tuple: train_generator, validation_generator
    """
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator
