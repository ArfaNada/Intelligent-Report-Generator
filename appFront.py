import streamlit as st
from PIL import Image
import requests
import io

st.set_page_config(page_title="Intelligent Report Generator", page_icon="🧾", layout="wide")

# --- Title ---
st.markdown(
    """
    <h1 style='text-align: center; color: #4B9CD3;'>🧾 Intelligent Report Generator</h1>
    <p style='text-align: center; color: gray;'>Upload an image and generate an intelligent report.</p>
    """,
    unsafe_allow_html=True
)

# --- Initialize session state ---
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False
if "product_info" not in st.session_state:
    st.session_state.product_info = None
if "report_data" not in st.session_state:
    st.session_state.report_data = None

# --- File Upload Section ---
if st.session_state.uploaded_file is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.rerun()

else:
    left_col, right_col = st.columns([1, 1])

    # Left column: Image display
    with left_col:
        image = Image.open(st.session_state.uploaded_file)
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        if st.button("🔁 Upload Another Image"):
            st.session_state.uploaded_file = None
            st.session_state.report_generated = False
            st.session_state.product_info = None
            st.session_state.report_data = None
            st.experimental_rerun()

    # Right column: Product + report
    with right_col:
        st.markdown("### ⚙️ Product Identification & Report")

        # Single button triggers both prediction and report
        if not st.session_state.report_generated:
            if st.button("Generate Report"):
                st.session_state.report_generated = True
                with st.spinner("Analyzing image and generating report..."):
                    try:
                        uploaded = st.session_state.uploaded_file
                        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}

                        # 1️⃣ Call predict
                        predict_resp = requests.post("http://192.168.1.18:5000/predict", files=files, timeout=60)
                        if predict_resp.status_code != 200:
                            st.error(f"❌ Error in product identification: {predict_resp.text}")
                            st.stop()
                        st.session_state.product_info = predict_resp.json()
                        product = st.session_state.product_info

                        # 2️⃣ Call analyze with the identified brand
                        brand_name = product['brand']
                        analyze_resp = requests.post(
                            "http://192.168.1.18:5000/analyze",
                            files={"file": (uploaded.name, io.BytesIO(uploaded.getvalue()), uploaded.type)},
                            data={"brand": brand_name},
                            timeout=60
                        )
                        if analyze_resp.status_code != 200:
                            st.error(f"❌ Error generating report: {analyze_resp.text}")
                            st.stop()
                        st.session_state.report_data = analyze_resp.json()

                    except Exception as e:
                        st.error(f"❌ Could not connect to backend: {e}")
                        st.stop()

        # Display product info
        if st.session_state.product_info:
            product = st.session_state.product_info
            st.markdown("### 🧠 Identified Product")
            st.markdown(f"**Brand:** {product['brand']}")
            st.markdown(f"**Product Name:** {product['product_name']}")
            st.markdown(f"**Similarity Score:** {product['similarity']:.3f}")

        # Display report
        if st.session_state.report_data:
            report = st.session_state.report_data
            st.markdown("### 📊 Sentiment & Emotion Report")
            st.markdown(f"⭐ **Average Rating:** {report['average_rating']} / 5")
            st.write(f"**Positive:** {report['sentiment_percentages']['Positive']}%")
            st.write(f"**Negative:** {report['sentiment_percentages']['Negative']}%")
            st.markdown(f"**Dominant Emotion:** {report['dominant_emotion']}")
