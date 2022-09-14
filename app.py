# Import Library
import base64
import json
import requests
import streamlit as st

def predict_image(image):
    response = requests.post(
        url='https://pizza-vs-icecream.el.r.appspot.com/predict/',
        data=json.dumps({
            'image': base64.b64encode(image).decode('utf8')
        }),
        headers={
            'Content-type': 'application/json',
            'Accept': 'text/plain'
        }
    )
    return response
    
def main():
    st.title('Pizza 🍕 vs Ice-cream 🍨 Classifier')
    st.info(
        'The model is trained on the '
        '[Pizza vs Ice Cream](https://www.kaggle.com/datasets/hemendrasr/pizza-vs-ice-cream) '
        'dataset hosted on Kaggle.'
    )
    st.write(
        'Upload an image to predict if the image is of icecream or pizza.'
    )

    uploaded_image = st.file_uploader(
        label='Upload an image',
        type=['png', 'jpg', 'jpeg'],
        help="Tip: if you're on a mobile device you can also take a photo",
    )

    if uploaded_image is not None:
        st.header('Uploaded Image')
        st.image(
            uploaded_image, use_column_width=True
        )

        predict_button = st.button(
            label='Predict'
        )

        if predict_button:
            with st.spinner():
                response = predict_image(uploaded_image.read())
                if response.status_code == 200:
                    response = json.loads(response.content.decode('UTF-8'))
                    st.success(
                        f"The uploaded image is of **{response['label']}**"
                        f" with a probability of {round(float(response['probability'])*100, 2)}%"
                    )
                else:
                    st.error('Internal Server Error')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.exception(e)
