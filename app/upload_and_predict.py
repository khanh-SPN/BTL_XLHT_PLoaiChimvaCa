from flask import Flask, request, render_template
from predict import predict

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload_and_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file uploaded"
        file = request.files['file']
        if file.filename == '':
            return "No file selected"
        file_path = f"./static/{file.filename}"
        file.save(file_path)

        result = predict(file_path)
        return render_template("result.html", result=result, image_path=file_path)

    return render_template("upload.html")

def start_app():
    app.run(debug=True)
