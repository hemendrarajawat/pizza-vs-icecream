import numpy as np
from flask import Flask, request, jsonify
from utils import preprocess_img, load_model, STATUS, CONFIG


app = Flask(__name__)

model = load_model()

@app.route('/predict/', methods=['POST'])
def predict():
    try:
        if 'image' not in request.json:
            return (
                jsonify({
                    'status': 'error',
                    'message': 'Image is required.'
                }),
                STATUS.BAD_REQUEST
            )

        input = preprocess_img(request.json['image'])
        
        result = model.predict(input)[0]

        return (
            jsonify({
                'status': 'success',
                'label': CONFIG.CLASS_LABEL[np.argmax(result)],
                'probability': str(result.max())
            }),
            STATUS.SUCCESS
        )
    except Exception as e:
        return (
            jsonify({
                'status': 'error',
                'message': e
            }),
            STATUS.INTERNAL_ERROR
        )


if __name__ == "__main__":
    app.run(debug=False)