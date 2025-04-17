import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from model import DrowningDetectionCNN
import io
import sys
import types
sys.modules['cv2'] = types.SimpleNamespace()

def annotate_image(image: Image.Image, prediction: dict) -> Image.Image:
    draw = ImageDraw.Draw(image)
    
    x_center = prediction["x"]
    y_center = prediction["y"]
    width = prediction["width"]
    height = prediction["height"]
    confidence = prediction["confidence"]
    class_name = prediction["class"]

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

    label = f"{class_name} ({confidence:.2f})"
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((x1, y1 - 20), label, fill="green", font=font)

    return image

def main():
    st.title("Drowning Detection - Minor Project")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=True)

        with open("temp_uploaded.jpg", "wb") as f:
            f.write(file_bytes)

        model = DrowningDetectionCNN()
        results = model.predict(path="temp_uploaded.jpg")
        pred = results["predictions"]

        annotated_image = annotate_image(image.copy(), pred)
        st.image(annotated_image, caption="Annotated Image", use_column_width=True)

        st.success(f"Detected: {pred['class']} with {pred['confidence']:.2f} confidence")

if __name__ == "__main__":
    main()
