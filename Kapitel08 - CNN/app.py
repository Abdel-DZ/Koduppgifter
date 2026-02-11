import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model once
model = ResNet50(weights='imagenet')

st.title("Image Classification with ResNet50")

# File upload
img_paths = [ 
    r"C:\Users\Admin\Desktop\MAI24\Python_AI_Del 2 DL\ai_tillaempad_ml-main\Kunskapskontroll\pictures\20260123_212122.jpg", 
    r"C:\Users\Admin\Desktop\MAI24\Python_AI_Del 2 DL\ai_tillaempad_ml-main\Kunskapskontroll\pictures\20250914_174823.jpg" ,
    r"C:\Users\Admin\Desktop\MAI24\Python_AI_Del 2 DL\ai_tillaempad_ml-main\Kunskapskontroll\pictures\20250824_133024.jpg",
    r"C:\Users\Admin\Desktop\MAI24\Python_AI_Del 2 DL\ai_tillaempad_ml-main\Kunskapskontroll\pictures\20260123_230154.jpg" 
]

# Loop through images
for idx, path in enumerate(img_paths):
    st.subheader(f"Bild {idx+1}")

    # Load and show image
    img = Image.open(path)
    st.image(img, caption=f"Bild {idx+1}", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]

    # Display predictions
    st.write("Top 3 Predictions:")
    for pred in decoded_preds:
        st.write(f"{pred[1]}: {pred[2]*100:.2f}%")

