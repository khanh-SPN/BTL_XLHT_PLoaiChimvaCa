from tensorflow.keras.models import load_model
from data_preprocessing import create_data_generators

# Đường dẫn mô hình và dữ liệu kiểm tra
MODEL_PATH = "../models/best_model.h5"
TEST_DIR = "../data/test"

# Load mô hình
model = load_model(MODEL_PATH)

# Tạo generator cho dữ liệu kiểm tra
_, test_gen = create_data_generators(TEST_DIR)

# Đánh giá mô hình
results = model.evaluate(test_gen)
print("Loss:", results[0])
print("Accuracy:", results[1])
