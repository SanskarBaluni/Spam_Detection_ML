import string
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

ps = PorterStemmer()

st.markdown("""
    <style>
        .stApp {
        background: url("https://easydmarc.com/blog/wp-content/webp-express/webp-images/doc-root/wp-content/uploads/2022/03/Why-is-DMARC-Failing.jpg.webp");
        background-size: cover;
        }
    </style>""", unsafe_allow_html=True)


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
st.markdown("<h1 style='text-align: center;'>Email/SMS Spam Detector</h1>", unsafe_allow_html=True)
input_sms = st.text_area('', placeholder="Enter the sms/email here")
if st.button('PREDICT', use_container_width=True):
    # 1.preprocess
    transformed_sms = transform_text(input_sms)
    # 2.vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3.predict
    result = model.predict(vector_input)[0]
    # 4.Display
    if result == 1:
        st.markdown("<h1 style='text-align: center;'>SPAM</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: center;'>NOT SPAM</h1>", unsafe_allow_html=True)
