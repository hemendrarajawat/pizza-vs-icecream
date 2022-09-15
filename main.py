import numpy as np
from flask import Flask, request, jsonify
from utils import preprocess_img, load_model, STATUS, CONFIG


app = Flask(__name__)

model = load_model()

@app.route('/', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
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
        else:
            return (
                jsonify({
                    'status': 'success',
                    'message': 'API is working...'
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
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))