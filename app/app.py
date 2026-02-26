import streamlit as st
from inference.predict_video import predict

st.title("Emotion Suppression Detection")

video = st.file_uploader("Upload Video")

if video:

    with open("input.mp4", "wb") as f:
        f.write(video.read())

    score = predict("input.mp4")

    st.write("Suppression Score:", score)

    if score > 0.6:
        st.error("High Suppression Detected")
    else:
        st.success("Low Suppression")