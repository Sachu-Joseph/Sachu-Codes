import spacy
from textblob import TextBlob
import sqlite3
import re
import random
from datetime import datetime

# Load dataset
def load_dataset():
    df = pd.read_csv("/mnt/data/helpdesk_customer_tickets.csv")
    return df


# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("support_interactions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            customer_query TEXT,
            bot_response TEXT,
            sentiment TEXT,
            escalated BOOLEAN
        )
    """)
    conn.commit()
    conn.close()

# Log interaction to database
def log_interaction(query, response, sentiment, escalated=False):
    conn = sqlite3.connect("support_interactions.db")
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO interactions (timestamp, customer_query, bot_response, sentiment, escalated) VALUES (?, ?, ?, ?, ?)",
        (timestamp, query, response, sentiment, escalated)
    )
    conn.commit()
    conn.close()
# Find best match from past queries
def get_similar_response(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    index = similarity.argmax()
    if similarity[0, index] > 0.5:  # threshold for similarity
        return df.iloc[index]["response"]
    return None

# FAQ database (mock)
FAQ = {
    "hours": "Our support hours are 9 AM to 5 PM, Monday to Friday.",
    "shipping": "Shipping takes 3-5 business days for standard delivery.",
    "returns": "You can return items within 30 days of purchase. Please provide your order ID.",
    "contact": "Reach us at support@example.com or call 1-800-123-4567."
}

# Mock order database
ORDERS = {
    "ORD123": {"status": "Shipped", "delivery_date": "2025-05-10"},
    "ORD456": {"status": "Processing", "delivery_date": "2025-05-15"},
    "ORD789": {"status": "Delivered", "delivery_date": "2025-05-01"}
}

# Intent recognition
def detect_intent(query):
    doc = nlp(query.lower())
    if any(token.lemma_ in ["faq", "question", "help"] for token in doc):
        return "faq"
    elif any(token.lemma_ in ["order", "track", "status"] for token in doc):
        return "order_status"
    elif any(token.lemma_ in ["refund", "return", "cancel"] for token in doc):
        return "refund"
    elif any(token.lemma_ in ["human", "agent", "escalate"] for token in doc):
        return "escalate"
    else:
        return "general"

# Extract order ID from query
def extract_order_id(query):
    match = re.search(r"\b(ORD\d{3})\b", query, re.IGNORECASE)
    return match.group(1) if match else None

# Sentiment analysis
def analyze_sentiment(query):
    blob = TextBlob(query)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

# Handle FAQ queries
def handle_faq(query):
    query_lower = query.lower()
    for key, answer in FAQ.items():
        if key in query_lower:
            return answer
    return "I'm not sure about that. Could you clarify or ask something else?"

# Handle order status
def handle_order_status(query):
    order_id = extract_order_id(query)
    if order_id and order_id in ORDERS:
        order = ORDERS[order_id]
        return f"Order {order_id} is {order['status']}. Estimated delivery: {order['delivery_date']}."
    return "Please provide a valid order ID (e.g., ORD123)."

# Handle refund requests
def handle_refund(query):
    order_id = extract_order_id(query)
    if order_id and order_id in ORDERS:
        if ORDERS[order_id]["status"] == "Delivered":
            return f"Refund request for {order_id} initiated. You'll hear back within 48 hours."
        else:
            return f"Cannot process refund for {order_id}. Order is still {ORDERS[order_id]['status']}."
    return "Please provide a valid order ID to process a refund."

# Escalate to human agent
def escalate_to_agent(query, sentiment):
    if sentiment == "negative":
        return "I'm sorry you're frustrated. Connecting you to a human agent now."
    return "Escalating your query to a human agent. Please wait."

# Generate response based on intent
def generate_response(query):
    sentiment = analyze_sentiment(query)
    intent = detect_intent(query)
    
    # Tailor response tone based on sentiment
    greeting = {
        "positive": "Happy to help! ",
        "negative": "I'm sorry you're having trouble. ",
        "neutral": "Hello! "
    }[sentiment]
    
    if intent == "faq":
        response = handle_faq(query)
    elif intent == "order_status":
        response = handle_order_status(query)
    elif intent == "refund":
        response = handle_refund(query)
    elif intent == "escalate" or sentiment == "negative":
        response = escalate_to_agent(query, sentiment)
        log_interaction(query, response, sentiment, escalated=True)
        return greeting + response
    else:
        response = "Could you clarify your request? I'm here to assist with FAQs, order tracking, refunds, or escalation."
    
    log_interaction(query, response, sentiment)
    return greeting + response

# Main chatbot loop
def main():
    init_db()
    print("Welcome to Customer Support Chatbot! Type 'exit' to quit.")
    
    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        if not query:
            print("Please enter a query.")
            continue
        
        try:
            response = generate_response(query)
            print(f"Bot: {response}")
        except Exception as e:
            print("Bot: Sorry, something went wrong. Please try again.")
            log_interaction(query, f"Error: {str(e)}", "neutral")

if __name__ == "__main__":
    main()
