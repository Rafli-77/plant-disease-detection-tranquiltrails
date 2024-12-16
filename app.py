import streamlit as st
from streamlit_option_menu import option_menu
from ultralytics import YOLO
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import cv2
import numpy as np

# Load YOLO models
paddy_model = YOLO("best1.pt")
chili_model = YOLO("best2.pt")
onion_model = YOLO("best3.pt")

# Streamlit page configuration
st.set_page_config(
    page_title="Tranquil Trails - Deteksi Objek dan Chatbot",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS for green leaf background
st.markdown(
    """
    <style>
        body {
            background-color: #228B22; /* Hijau daun */
            color: white; /* Teks menjadi putih */
        }
        [data-testid="stSidebar"] {
            background-color: #2E8B57; /* Hijau gelap untuk sidebar */
        }
        .css-1v3fvcr {
            border: 2px dashed white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load GPT-2 model and tokenizer
@st.cache_resource
def load_gpt2_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

gpt2_model, gpt2_tokenizer = load_gpt2_model()

def generate_gpt2_response(question, model, tokenizer):
    inputs = tokenizer.encode(question, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Navbar
selected = option_menu(
    menu_title=None,  # required
    options=["Home", "Login", "Register", "About Us"],  # menu options
    icons=["house", "box-arrow-in-right", "person-plus", "info-circle"],  # icons for menu options
    menu_icon="cast",  # main menu icon
    default_index=0,  # default active menu
    orientation="horizontal",  # horizontal navbar
    styles={
        "container": {"padding": "0!important", "background-color": "#228B22"},
        "icon": {"color": "white", "font-size": "18px"},
        "nav-link": {"color": "white", "font-size": "18px", "text-align": "center", "margin": "0px"},
        "nav-link-selected": {"background-color": "#2E8B57"},
    }
)

if selected == "Home":
    # Sidebar Navigation
    st.sidebar.title("Tranquil Trails")
    st.sidebar.header("Pilih Fitur")
    feature_choice = st.sidebar.radio(
        "Pilih fitur yang ingin digunakan:",
        ("Deteksi Tanaman", "Chatbot")
    )

    if feature_choice == "Deteksi Tanaman":
        # Pilihan model deteksi
        st.sidebar.header("Pilih Model")
        model_choice = st.sidebar.radio(
            "Pilih jenis tanaman yang ingin dideteksi:",
            ("Padi", "Cabai", "Bawang")
        )

        # Map pilihan ke model
        if model_choice == "Padi":
            selected_model = paddy_model
        elif model_choice == "Cabai":
            selected_model = chili_model
        else:
            selected_model = onion_model

        # Main content layout
        st.title("🌾 Deteksi Objek untuk Tanaman")
        st.markdown(
            """
            <p style="font-size: 16px; text-align: center;">
                Pilih model deteksi di menu sebelah kiri, unggah gambar, dan lihat hasil deteksi.
            </p>
            """,
            unsafe_allow_html=True
        )

        # Check if a model is selected
        st.markdown(f"### 🌱 Model Terpilih: **{model_choice}**")

        # Upload and process image
        uploaded_file = st.file_uploader(
            f"Unggah gambar {model_choice} untuk memulai deteksi:",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            # Convert uploaded file to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Display uploaded image
            st.image(img, channels="BGR", caption="📷 Gambar yang diunggah", use_column_width=True)

            # Run selected YOLO model on the image
            st.markdown("### 🔄 Memproses gambar dengan model yang dipilih...")
            results = selected_model(img)

            # Display results
            st.markdown("### ✅ Hasil Deteksi")
            annotated_image = results[0].plot()
            st.image(annotated_image, caption="🔍 Deteksi Objek", use_column_width=True)
        else:
            st.markdown(
                """
                <div style="border: 2px dashed #228B22; padding: 20px; text-align: center; border-radius: 10px;">
                    <h3 style="color: #228B22;">Menunggu gambar diunggah</h3>
                    <p style="color: #555;">Pilih dan unggah gambar untuk mendeteksi objek</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    elif feature_choice == "Chatbot":
        # Chatbot Interface
        st.title("🤖 Chatbot AI")
        st.markdown(
            """
            <p style="font-size: 16px; text-align: center;">
                Ajukan pertanyaan terkait pertanian atau penggunaan aplikasi ini.
            </p>
            """,
            unsafe_allow_html=True
        )

        # Input teks dari pengguna
        user_input = st.text_input("Ketik pertanyaan Anda di bawah ini:")

        if st.button("Kirim"):
            if user_input.strip():
                with st.spinner("Sedang membuat respons..."):
                    response = generate_gpt2_response(user_input, gpt2_model, gpt2_tokenizer)
                st.success("Chatbot AI Menjawab:")
                st.write(response)
            else:
                st.warning("Silakan masukkan pertanyaan terlebih dahulu.")

elif selected == "Login":
    st.title("🔑 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username and password:
            st.success(f"Selamat datang, {username}!")
        else:
            st.error("Silakan masukkan username dan password yang valid.")

elif selected == "Register":
    st.title("📝 Register")
    new_username = st.text_input("Username Baru")
    new_password = st.text_input("Password Baru", type="password")
    confirm_password = st.text_input("Konfirmasi Password", type="password")
    if st.button("Register"):
        if new_password == confirm_password:
            st.success("Registrasi berhasil! Silakan login.")
        else:
            st.error("Password tidak cocok. Silakan coba lagi.")

elif selected == "About Us":
    st.title("ℹ️ About Us")
    st.markdown(
        """
        <p style="font-size: 18px; text-align: center;">
            <strong>Tranquil Trails</strong> adalah aplikasi berbasis AI yang membantu petani untuk mendeteksi tanaman
            dan menyediakan chatbot pintar untuk mendukung kebutuhan informasi agrikultur.
        </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)