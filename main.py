import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from predict import predict

# Tạo ứng dụng Tkinter
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Dự đoán Chim và Cá")
        self.root.geometry("500x500")
        
        # Nút tải ảnh
        self.upload_button = tk.Button(self.root, text="Tải Ảnh", command=self.upload_image)
        self.upload_button.pack(pady=20)
        
        # Khung hiển thị ảnh
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)
        
        # Hiển thị kết quả
        self.result_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.result_label.pack(pady=20)
        
    def upload_image(self):
        # Mở hộp thoại để chọn ảnh
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            # Hiển thị ảnh
            img = Image.open(file_path).resize((224, 224))  # Resize ảnh
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
            
            # Dự đoán
            result = predict(file_path)
            self.result_label.config(text=f"Kết quả: {result}")

# Chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
