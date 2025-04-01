from flask import Flask, request, jsonify, render_template
from inference_engine.runner import MolecularGenerator
app = Flask(__name__)
generator = MolecularGenerator(device="cpu")
@app.route('/')
def home():
    return render_template('index.html')

# model = MyModel()
# model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
# model.eval()

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            user_input = data.get("user_input", "").strip()

            if not user_input:
                return jsonify({"error": "Input data is required"}), 400

            # 调用封装好的模型方法
            prediction = generator(user_input)

            return jsonify({"prediction": prediction})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template("predict.html")


@app.route('/molecule', methods=['GET', 'POST'])
def molecule():
    if request.method == 'POST':
        try:
            data = request.get_json()
            input_data = data.get("input_data", "").strip()
            num_samples = int(data.get("num_samples", 16))
            batch_size = int(data.get("batch_size", 8))

            if not input_data:
                return jsonify({"error": "Input data is required"}), 400

            # 调用封装的生成分子方法
            molecules = generator.generate_molecules(input_data, num_samples=num_samples, batch_size=batch_size)

            return jsonify({"molecules": molecules})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template("molecule.html")

if __name__ == "__main__":
    app.run(debug=True)
