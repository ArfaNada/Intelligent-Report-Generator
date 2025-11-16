import streamlit as st
from PIL import Image

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

# --- File Upload Section ---
if st.session_state.uploaded_file is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.rerun()

else:
    # --- Side-by-side layout ---
    left_col, right_col = st.columns([1, 1])

    # Left column: Image
    with left_col:
        image = Image.open(st.session_state.uploaded_file)
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        # Upload Another Image button
        btn_col1, btn_col2 = st.columns([1, 3])
        with btn_col2:
            if st.button("🔁 Upload Another Image"):
                st.session_state.uploaded_file = None
                st.session_state.report_generated = False
                st.rerun()

    # Right column: Report generation
    with right_col:
        st.markdown("### ⚙️ Generate Report")

        # Show button only if report not yet generated
        if not st.session_state.report_generated:
            if st.button("Generate Report"):
                st.session_state.report_generated = True

        # Display report once generated
        if st.session_state.report_generated:
            with st.spinner("Analyzing image and generating report..."):
                st.success("✅ Report generated successfully!")

                st.markdown("### 🧠 Generated Report")

                # Section 1: Top 3 Features
                st.markdown("#### 1️⃣ Top 3 Features")
                st.markdown(
                    "- Feature 1: Placeholder\n"
                    "- Feature 2: Placeholder\n"
                    "- Feature 3: Placeholder"
                )

                # Section 2: Sentiment Overview
                st.markdown("#### 2️⃣ Sentiment Overview")
                st.markdown(
                    "Overall sentiment: Placeholder\n"
                    "Positive: Placeholder\n"
                    "Neutral: Placeholder\n"
                    "Negative: Placeholder"
                )

                # Section 3: Recommended Usage
                st.markdown("#### 3️⃣ Recommended Usage")
                st.markdown(
                    "- Placeholder usage recommendation 1\n"
                    "- Placeholder usage recommendation 2\n"
                    "- Placeholder usage recommendation 3"
                )

                # Section 4: Brand Summary
                st.markdown("#### 4️⃣ Brand Summary")
                st.markdown(
                    "Brand Name: Placeholder\n"
                    "Brand Reputation: Placeholder\n"
                    "Other relevant info: Placeholder"
                )
