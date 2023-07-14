import string
import nltk
nltk.download('stopwords')
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

ps = PorterStemmer()


st.markdown("""
    <style>
        .stApp {
        background: url("https://postaga.com/wp-content/webpc-passthru.php?src=https://postaga.com/wp-content/uploads/2021/12/spam-guide-cover-1536x1024.jpg&nocache=1");
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
st.title("     _____Email/SMS Spam Detector     ")
input_sms = st.text_area('', placeholder="Enter the sms/email here")
if st.button('PREDICT'):
    # 1.preprocess
    transformed_sms = transform_text(input_sms)
    # 2.vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3.predict
    result = model.predict(vector_input)[0]
    # 4.Display
    if result == 1:
        st.header("**Spam**")
    else:
        st.header("**Not Spam**")
