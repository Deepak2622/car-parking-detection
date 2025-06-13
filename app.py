import streamlit as st
import cv2
import numpy as np
import tempfile
from fusion_module import get_fused_predictions
import pandas as pd
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Car Parking Detection", layout="centered")
st.title("🚗 Car Parking Detection (YOLOv8m + Mask R-CNN)")
st.caption("Upload a parking lot image and get slot status like a boss 😎")

uploaded_file = st.file_uploader("Upload Parking Lot Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save the uploaded image to a specific file
    save_path = Path("fusion_output/uploaded_image.jpg")
    save_path.parent.mkdir(parents=True, exist_ok=True)  # make sure folder exists

    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    # Call your fusion prediction
    preds, annotated_img = get_fused_predictions(str(save_path), return_annotated=True)

    # Count parking slots
    occupied = preds.count("occupied")
    empty = preds.count("empty")
    total = occupied + empty

    # Show result image
    st.image(annotated_img, caption="Fused Detection Output", use_column_width=True)

    # Show result summary
    st.markdown("### 📊 Parking Summary")
    st.write(f"**Total Slots:** {total}")
    st.write(f"✅ Empty Slots:** {empty}")
    st.write(f"🚫 Occupied Slots:** {occupied}")

    # Allow CSV download
    df = pd.DataFrame([[Path(uploaded_file.name).name, total, occupied, empty]],
                      columns=["Image", "Total", "Occupied", "Empty"])
    csv_file = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv_file, file_name="slot_summary.csv", mime="text/csv")

    st.success("All done! Now go impress the world 🌍✨")
