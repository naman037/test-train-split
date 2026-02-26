import streamlit as st
import joblib
import requests
from bs4 import BeautifulSoup
model = joblib.load('fake_news_model.pkl')
counter = joblib.load('word_counter.pkl')
st.title("🚨 Fake News Detector Pro")
st.write("Paste a news article link below, and the AI will scrape the headline and analyze it!")
user_url = st.text_input("Enter Article URL:")
if st.button("Analyze Link"):
    if user_url:
        st.info("Scraping website... 🕵️‍♂️")
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            website_data = requests.get(user_url, headers=headers)
            soup = BeautifulSoup(website_data.text, 'html.parser')
            scraped_headline = soup.find('title').text
            
            st.write(f"**Found Headline:** {scraped_headline}")
            test_math = counter.transform([scraped_headline])
            prediction = model.predict(test_math)
            probabilities = model.predict_proba(test_math)[0]
            confidence = max(probabilities) * 100
            if prediction[0] == 1:
                st.success(f"VERDICT: REAL News 📰 (Confidence: {confidence:.2f}%)")
            else:
                st.error(f"VERDICT: FAKE News 🚨 (Confidence: {confidence:.2f}%)")
                
        except Exception as e:
            st.warning("Anti-Bot Security blocked us from reading that site! Try a different link.")
    else:
        st.warning("Please enter a URL first!")