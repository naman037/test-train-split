import joblib

print("Waking up the AI... 🧠")

model = joblib.load('fake_news_model.pkl')
counter = joblib.load('word_counter.pkl')

print("AI is awake and ready in 0.1 seconds!\n")
print("-" * 40)


while True:
    user_headline = input("Enter a headline (or type 'quit' to exit): ")
    
    if user_headline.lower() == 'quit':
        print("Shutting down AI. Goodbye!")
        break
    test_math = counter.transform([user_headline])
    

    prediction = model.predict(test_math)

    prediction = model.predict(test_math)
    probabilities = model.predict_proba(test_math)[0]
    confidence = max(probabilities) * 100
    
    if prediction[0] == 1:
        print(f"➡️  VERDICT: REAL News 📰 (Confidence: {confidence:.2f}%)\n")
    else:
        print(f"➡️  VERDICT: FAKE News 🚨 (Confidence: {confidence:.2f}%)\n")
    

    if prediction[0] == 1:
        print("➡️  VERDICT: REAL News 📰\n")
    else:
        print("➡️  VERDICT: FAKE News 🚨\n")