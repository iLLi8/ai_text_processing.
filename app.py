from flask import Flask, render_template, request
from text_processor import process_text

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    processed_text = None
    if request.method == "POST":
        input_text = request.form.get("input_text", "")
        print(f"Input Text: {input_text}")  # Debug log
        processed_text = process_text(input_text)
        print(f"Processed Text: {processed_text}")  # Debug log
    return render_template("index.html", processed_text=processed_text)

if __name__ == "__main__":
    app.run(debug=True)

