import streamlit as st
import pickle
import re
import torch
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from model import LogisticRegression  # Ensure this matches your model's definition
from model import Train_Test_Model  # Ensure this matches your model's definition

# Read Markdown file
def read_markdown_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
markdown_content = read_markdown_file(r"D:\Code Playground\Solving Research Paper\Data_sets\Logistic_Regression_Spam.md")

# Load the trained model and vectorizer
with open(r"D:\Code Playground\Solving Research Paper\Logistic Regression\model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open(r"D:\Code Playground\Solving Research Paper\Logistic Regression\vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to preprocess email text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

# Set page title and layout
st.set_page_config(page_title="Spam Classifier & Project Info", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
        :root {
            --primary-color: #d98f57;
        }
        body {
            font-family: 'Quicksand', sans-serif;
            color: #2c3e50;
        }
        .header-title {
            text-align: center;
            color: var(--primary-color);
            font-family: 'Times New Roman', Times, serif;
            font-size: 2.5em;
            text-transform: uppercase;
        }
        .section {
            background: white;
            padding: 30px;
            margin: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            border-left: 5px solid var(--primary-color);
        }
        .section h2 {
            color: #004d40;
            font-size: 2em;
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation Sidebar
st.sidebar.title("Navigation")
sections = ["Spam Classifier", "Libraries Used", "Logistic Regression Model", "Model Workflow", "Future Implementation", "Contact"]
selected_section = st.sidebar.radio("Go to", sections)

# Page Header
st.markdown("<h1 class='header-title'>Spam Classifier & Project Information</h1>", unsafe_allow_html=True)

# Spam Classification Section
if selected_section == "Spam Classifier":
    st.header("Email Spam Classifier")
    st.write("Enter an email message to classify it as Spam or Not Spam.")
    email_text = st.text_area("Enter Email Content", height=200)
    if st.button("Classify"):
        if email_text.strip():
            processed_text = preprocess_text(email_text)
            vectorized_text = vectorizer.transform([processed_text])
            if hasattr(vectorized_text[0], "toarray"):
                vectorized_text = torch.tensor(vectorized_text[0].toarray(), dtype=torch.float32)
            else:
                vectorized_text = torch.tensor(vectorized_text[0], dtype=torch.float32)
            prediction = Train_Test_Model.predict(model, vectorized_text)
            result = "Spam" if prediction == 1 else "Not Spam"
            st.subheader(f"Result: {result}")
        else:
            st.warning("Please enter email content.")

# Other Informational Sections
elif selected_section == "Libraries Used":
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("Libraries Used")
    st.write("- **PyTorch:** For building and training the logistic regression model.")
    st.write("- **NumPy:** For numerical computations and data manipulation.")
    st.write("- **Scikit-learn:** For text preprocessing and feature extraction using `CountVectorizer` and `TfidfTransformer`.")
    st.write("- **Pandas:** For dataset handling and preparation.")
    st.markdown("</div>", unsafe_allow_html=True)

elif selected_section == "Logistic Regression Model":
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("Logistic Regression Model")
    st.write("The spam detector relies on a custom logistic regression model implemented from scratch using PyTorch.")
    st.write("**Key Details:**")
    st.write("- **Model Type:** Binary classification (Spam vs. Not Spam).")
    st.write("- **Feature Extraction:** Text data is transformed into numerical features using `CountVectorizer` and `TfidfTransformer`.")
    st.write("- **Training Dataset:** The model is trained on the Kaggle SMS Spam Collection Dataset, containing labeled spam and ham messages.")
    st.markdown("</div>", unsafe_allow_html=True)

elif selected_section == "Model Workflow":
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("Model Workflow")
    st.markdown(markdown_content, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected_section == "Future Implementation":
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("Future Implementation")
    st.write("Upcoming enhancements and features planned for this project...")
    st.markdown("</div>", unsafe_allow_html=True)

elif selected_section == "Contact":
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("Contact")
    st.write("For any queries, please reach out to us via email or social media.")
    st.markdown("Email: [himanshusr451tehs@gmail.com](mailto:himanshusr451tehs@gmail.com) | GitHub: [Himanshu7921](https://github.com/Himanshu7921/Himanshu7921)")
    st.markdown("</div>", unsafe_allow_html=True)


