import streamlit as st
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('stopwords')
# -------------------- PAGE SETTINGS --------------------
st.set_page_config(page_title="AI FAQ Chatbot", page_icon="🤖")

st.title("🤖 AI FAQ Chatbot")
st.write("Ask any Python or AI related question!")

# -------------------- FAQ DATA --------------------
questions = [
    "What is Python?",
    "Who developed Python?",
    "What is Artificial Intelligence?",
    "What is Machine Learning?",
    "What is Deep Learning?",
    "What is Streamlit?",
    "What is NLP?",
    "What is Data Science?"
]

answers = [
    "Python is a popular high-level programming language.",
    "Python was developed by Guido van Rossum.",
    "Artificial Intelligence enables machines to think and perform tasks like humans.",
    "Machine Learning is a subset of AI that learns from data.",
    "Deep Learning is a subset of Machine Learning based on neural networks.",
    "Streamlit is a Python framework used for building web applications.",
    "NLP stands for Natural Language Processing. It helps computers understand human language.",
    "Data Science is the process of extracting useful insights from data."
]

# -------------------- TEXT PREPROCESSING --------------------
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    words = word_tokenize(text)

    cleaned_words = []

    for word in words:
        if word not in stop_words and word not in string.punctuation:
            cleaned_words.append(word)

    return " ".join(cleaned_words)

processed_questions = [preprocess(q) for q in questions]

# -------------------- USER INPUT --------------------
user_question = st.text_input("Enter your question")

if st.button("Get Answer"):

    if user_question.strip() == "":
        st.warning("Please enter a question.")

    else:

        processed_user = preprocess(user_question)

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(processed_questions + [processed_user])

        similarity = cosine_similarity(vectors[-1], vectors[:-1])

        best_match = similarity.argmax()

        score = similarity[0][best_match]

        if score > 0.25:
            st.success("Answer:")
            st.write(answers[best_match])

            st.info(f"Similarity Score : {score:.2f}")

        else:
            st.error("Sorry! I couldn't find a matching answer.")