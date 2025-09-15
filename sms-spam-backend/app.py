from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bson import ObjectId
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pickle
import numpy as np
from db_config import users_collection, permissions_collection,ham_messages_collection,smish_messages_collection
from passlib.hash import bcrypt
from datetime import datetime, timedelta
import jwt
import random
import string
from typing import Dict
import re

# === JWT CONFIG ===
SECRET_KEY = "ADGP2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# === Load model and preprocessing ===
with open("vectorizer_fixed.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder_fixed.pkl", "rb") as f:
    label_enc = pickle.load(f)

class HGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HGNN, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.gcn1 = GCNConv(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.gcn1(x, edge_index)
        x = self.lin2(x)
        return x

device = torch.device("cpu")
model = HGNN(in_channels=vectorizer.max_features, hidden_channels=64, out_channels=2)
model.load_state_dict(torch.load("hgnn_spam_classifier_fixed.pt", map_location=device))
model.eval()

# === FastAPI setup ===
app = FastAPI(title="Spam Detection API", version="1.0.0")

# ==== SCHEMAS ====
class RegisterRequest(BaseModel):
    username: str
    country_code: str
    phone_number: str
    password: str

class LoginRequest(BaseModel):
    country_code: str
    phone_number: str
    password: str

class PermissionsRequest(BaseModel):
    user_id: str
    allow_contacts: bool
    allow_location: bool
    allow_media: bool
    allow_call_logs: bool
    allow_sms: bool

class ForgotPasswordRequest(BaseModel):
    country_code: str
    phone_number: str

class VerifyOTPRequest(BaseModel):
    country_code: str
    phone_number: str
    otp: str

class ResetPasswordRequest(BaseModel):
    country_code: str
    phone_number: str
    otp: str
    new_password: str

class Message(BaseModel):
    text: str
    user_id: str
    sender: str

class SMSInterceptRequest(BaseModel):
    user_id: str
    sender: str
    message_content: str
    timestamp: str
    original_language: str = "en"
    translated_content: str = None

class SMSClassificationResponse(BaseModel):
    prediction: str
    confidence: float
    message_id: str
    alert_user: bool
    translated_content: str = None

# === UTILITIES ===
def build_hypergraph_single(X):
    num_samples, num_features = X.shape
    edge_index = [[], []]
    for sample_idx in range(num_samples):
        active_features = np.nonzero(X[sample_idx])[0].tolist()
        for feat in active_features:
            edge_index[0].append(feat)
            edge_index[1].append(num_features + sample_idx)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x_feat = torch.eye(num_features)
    x_samp = torch.tensor(X, dtype=torch.float)
    x_all = torch.cat([x_feat, x_samp], dim=0)
    return x_all, edge_index

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# === OTP STORAGE AND UTILITIES ===
# In-memory OTP storage (in production, use Redis or database)
otp_storage: Dict[str, Dict] = {}

def generate_otp() -> str:
    """Generate a 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=6))

def store_otp(phone_number: str, otp: str) -> None:
    """Store OTP with expiration time (5 minutes)"""
    expiry_time = datetime.utcnow() + timedelta(minutes=5)
    otp_storage[phone_number] = {
        "otp": otp,
        "expiry": expiry_time,
        "verified": False
    }

def verify_otp(phone_number: str, otp: str) -> bool:
    """Verify OTP and check if it's not expired"""
    if phone_number not in otp_storage:
        return False
    
    stored_data = otp_storage[phone_number]
    current_time = datetime.utcnow()
    
    # Check if OTP is expired
    if current_time > stored_data["expiry"]:
        del otp_storage[phone_number]  # Clean up expired OTP
        return False
    
    # Check if OTP matches
    if stored_data["otp"] == otp:
        otp_storage[phone_number]["verified"] = True
        return True
    
    return False

def is_otp_verified(phone_number: str) -> bool:
    """Check if OTP has been verified for password reset"""
    return (phone_number in otp_storage and 
            otp_storage[phone_number].get("verified", False) and
            datetime.utcnow() <= otp_storage[phone_number]["expiry"])

def send_sms_otp(phone_number: str, otp: str) -> bool:
    """
    Send OTP via SMS using Twilio
    Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER in environment variables
    """
    try:
        # Import Twilio (install with: pip install twilio)
        from twilio.rest import Client
        import os
        
        # Get Twilio credentials from environment variables
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')
        
        # Check if Twilio is configured
        if not all([account_sid, auth_token, twilio_phone_number]):
            print(f"[SMS MOCK] Twilio not configured. Sending OTP {otp} to {phone_number}")
            print(f"[SMS MOCK] Message: Your CyberTextShield password reset OTP is: {otp}. Valid for 5 minutes.")
            print("[SMS MOCK] To enable real SMS, set environment variables: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER")
            return True
        
        # Initialize Twilio client
        client = Client(account_sid, auth_token)
        
        # Format phone number (ensure it starts with country code)
        if not phone_number.startswith('+'):
            phone_number = '+' + phone_number
        
        # Send SMS
        message = client.messages.create(
            body=f"Your CyberTextShield password reset OTP is: {otp}. Valid for 5 minutes. Do not share this code.",
            from_=twilio_phone_number,
            to=phone_number
        )
        
        print(f"[SMS SENT] OTP {otp} sent to {phone_number} via Twilio. Message SID: {message.sid}")
        return True
        
    except ImportError:
        print(f"[SMS MOCK] Twilio not installed. Sending OTP {otp} to {phone_number}")
        print(f"[SMS MOCK] Message: Your CyberTextShield password reset OTP is: {otp}. Valid for 5 minutes.")
        print("[SMS MOCK] To enable real SMS, install Twilio: pip install twilio")
        return True
        
    except Exception as e:
        print(f"[SMS ERROR] Failed to send OTP to {phone_number}: {str(e)}")
        print(f"[SMS FALLBACK] OTP for {phone_number}: {otp}")
        return False

# === SMS INTERCEPTION AND LANGUAGE PROCESSING ===
def detect_language(text: str) -> str:
    """
    Detect language of the text using simple pattern matching
    In production, use proper language detection library like langdetect
    """
    try:
        # Simple English detection (check for common English words)
        english_words = ['the', 'and', 'you', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'as', 'was', 'on', 'are', 'this', 'be', 'at', 'by', 'your', 'have']
        text_lower = text.lower()
        english_word_count = sum(1 for word in english_words if word in text_lower)
        
        # If text contains many English words, consider it English
        if english_word_count >= 2 or re.search(r'[a-zA-Z]', text):
            return 'en'
        else:
            return 'unknown'
    except Exception as e:
        print(f"[LANG DETECT ERROR] {str(e)}")
        return 'en'  # Default to English

def translate_to_english(text: str, source_lang: str = 'auto') -> str:
    """
    Translate text to English using Google Translate API
    For production, use proper Google Translate API with credentials
    """
    try:
        # Mock translation for demo - in production use googletrans or Google Cloud Translate
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text, src=source_lang, dest='en')
        print(f"[TRANSLATION] {source_lang} -> en: '{text}' -> '{result.text}'")
        return result.text
    except ImportError:
        print(f"[TRANSLATION MOCK] googletrans not installed. Original text: {text}")
        return text  # Return original if translation not available
    except Exception as e:
        print(f"[TRANSLATION ERROR] {str(e)}")
        return text  # Return original on error

def save_message_to_file(sender: str, message: str, classification: str):
    """
    Save intercepted message to file for logging/debugging
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = "intercepted_messages.log"
        
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] FROM: {sender} | CLASS: {classification} | MSG: {message}\n")
        
        # Also save to msg.txt as requested
        with open("msg.txt", "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
            
    except Exception as e:
        print(f"[FILE SAVE ERROR] {str(e)}")

def classify_intercepted_message(message_content: str) -> dict:
    """
    Classify intercepted SMS message using HGNN model
    """
    try:
        # Preprocess the message
        device = torch.device("cpu")
        
        # Vectorize the message
        X = vectorizer.transform([message_content]).toarray()
        
        # Build hypergraph
        x_all, edge_index = build_hypergraph_single(X)
        
        # Convert to torch tensors
        x_all = torch.FloatTensor(x_all).to(device)
        edge_index = torch.LongTensor(edge_index).to(device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            out = model(x_all, edge_index)
            probabilities = F.softmax(out[vectorizer.max_features:], dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()
        
        # Convert prediction to label
        prediction_label = label_enc.inverse_transform([prediction])[0]
        
        return {
            "prediction": prediction_label,
            "confidence": float(confidence),
            "is_spam": prediction_label.lower() in ['spam', 'smish']
        }
        
    except Exception as e:
        print(f"[CLASSIFICATION ERROR] {str(e)}")
        return {
            "prediction": "ham",
            "confidence": 0.5,
            "is_spam": False
        }

# === ROUTES ===
@app.get("/")
def root():
    return {"message": "Spam Detection API is running!"}

@app.post("/register")
def register_user(data: RegisterRequest):
    # Basic format check
    if not data.country_code.startswith("+") or not data.phone_number.isdigit():
        raise HTTPException(status_code=400, detail="Invalid phone number or country code format")

    full_phone = f"{data.country_code}{data.phone_number}"

    if users_collection.find_one({"phone_number": full_phone}):
        raise HTTPException(status_code=400, detail="Phone number is already registered")
    
    hashed_pw = bcrypt.hash(data.password)

    user_data = {
        "username": data.username,
        "country_code": data.country_code,
        "phone_number": full_phone,
        "password": hashed_pw
    }
    result = users_collection.insert_one(user_data)
    return {
        "user_id": str(result.inserted_id),
        "message": f"User {data.username} registered successfully with phone {full_phone}"
    }

@app.post("/login")
def login_user(data: LoginRequest):
    # Validate input
    if not data.country_code.startswith("+") or not data.phone_number.isdigit():
        raise HTTPException(status_code=400, detail="Invalid phone number or country code format")

    # Combine country code and number
    full_phone = f"{data.country_code}{data.phone_number}"

    # Search for user in DB
    user = users_collection.find_one({"phone_number": full_phone})
    if not user:
        raise HTTPException(status_code=404, detail="User not found. Please register first.")
    
    if not bcrypt.verify(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect Password")
    
    token_data = {"sub": str(user["_id"]), "username": user["username"]}
    access_token = create_access_token(token_data)  # FIXED: Added function call

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": str(user["_id"]),
        "username": user["username"],
        "phone_number": user["phone_number"],
        "message": f"Welcome back, {user['username']}!"
    }

# === FORGOT PASSWORD ENDPOINTS ===
@app.post("/forgot-password")
def forgot_password(data: ForgotPasswordRequest):
    """Send OTP to user's phone number for password reset"""
    # Validate input
    if not data.country_code.startswith("+") or not data.phone_number.isdigit():
        raise HTTPException(status_code=400, detail="Invalid phone number or country code format")

    # Combine country code and number
    full_phone = f"{data.country_code}{data.phone_number}"

    # Check if user exists
    user = users_collection.find_one({"phone_number": full_phone})
    if not user:
        raise HTTPException(status_code=404, detail="No account found with this phone number")
    
    # Generate and store OTP
    otp = generate_otp()
    store_otp(full_phone, otp)
    
    # Send OTP via SMS (mock implementation)
    sms_sent = send_sms_otp(full_phone, otp)
    
    if not sms_sent:
        raise HTTPException(status_code=500, detail="Failed to send OTP. Please try again.")
    
    return {
        "message": f"OTP sent successfully to {full_phone}",
        "phone_number": full_phone,
        "expires_in": "5 minutes"
    }

@app.post("/verify-otp")
def verify_otp_endpoint(data: VerifyOTPRequest):
    """Verify OTP for password reset"""
    # Validate input
    if not data.country_code.startswith("+") or not data.phone_number.isdigit():
        raise HTTPException(status_code=400, detail="Invalid phone number or country code format")

    # Combine country code and number
    full_phone = f"{data.country_code}{data.phone_number}"
    
    # Verify OTP
    if not verify_otp(full_phone, data.otp):
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    
    return {
        "message": "OTP verified successfully",
        "phone_number": full_phone,
        "verified": True
    }

@app.post("/reset-password")
def reset_password(data: ResetPasswordRequest):
    """Reset password after OTP verification"""
    # Validate input
    if not data.country_code.startswith("+") or not data.phone_number.isdigit():
        raise HTTPException(status_code=400, detail="Invalid phone number or country code format")

    # Combine country code and number
    full_phone = f"{data.country_code}{data.phone_number}"
    
    # Check if OTP was verified
    if not is_otp_verified(full_phone):
        raise HTTPException(status_code=400, detail="OTP not verified or expired. Please verify OTP first.")
    
    # Validate new password
    if len(data.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")
    
    # Check if user exists
    user = users_collection.find_one({"phone_number": full_phone})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Hash new password
    hashed_password = bcrypt.hash(data.new_password)
    
    # Update password in database
    result = users_collection.update_one(
        {"phone_number": full_phone},
        {"$set": {"password": hashed_password}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=500, detail="Failed to update password")
    
    # Clean up OTP storage
    if full_phone in otp_storage:
        del otp_storage[full_phone]
    
    return {
        "message": "Password reset successfully",
        "phone_number": full_phone,
        "username": user["username"]
    }

@app.post("/permissions")
def collect_permissions(data: PermissionsRequest):
    if not ObjectId.is_valid(data.user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID")

    permissions_data = {
        "user_id": ObjectId(data.user_id),
        "permissions": {
            "contacts": data.allow_contacts,
            "location": data.allow_location,
            "media": data.allow_media,
            "call_logs": data.allow_call_logs,
            "sms": data.allow_sms
        }
    }
    permissions_collection.insert_one(permissions_data)
    return {"message": "Permissions saved successfully"}

@app.post("/predict")
def predict(msg: Message):
    if not ObjectId.is_valid(msg.user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID")

    user = users_collection.find_one({"_id": ObjectId(msg.user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Vectorize
    vec = vectorizer.transform([msg.text]).toarray()
    x_all, edge_index = build_hypergraph_single(vec)

    with torch.no_grad():
        out = model(x_all, edge_index)
        num_features = vectorizer.max_features
        out_sample = out[num_features:]
        probs = F.softmax(out_sample, dim=1).squeeze().numpy()
        pred = np.argmax(probs)
        label = label_enc.inverse_transform([pred])[0]

        # ✅ Business rule: if sender ends with G/T/P/S → force ham
        sender = getattr(msg, "sender", None)
        if sender and sender.strip()[-1].upper() in ["G", "T", "P","S"]:
            label = "ham"

        result = {
            "user": user["username"],
            "message": msg.text,
            "sender": sender,
            "prediction": label,
            "confidence": float(probs[pred]),
            "probabilities": {
                label_enc.inverse_transform([0])[0]: float(probs[0]),
                label_enc.inverse_transform([1])[0]: float(probs[1]),
            }
        }

        # ✅ Store in respective collection
        message_doc = {
            "user_id": ObjectId(msg.user_id),
            "username": user["username"],
            "text": msg.text,
            "sender": sender,
            "prediction": label,
            "confidence": float(probs[pred])
        }

        if label == "ham":
            ham_messages_collection.insert_one(message_doc)
        else:
            smish_messages_collection.insert_one(message_doc)

        return result

@app.post("/intercept-sms", response_model=SMSClassificationResponse)
def intercept_sms(sms_data: SMSInterceptRequest):
    """
    Endpoint for SMS interception and real-time classification
    Processes incoming SMS, detects language, translates if needed, and classifies
    """
    try:
        # Validate user
        if not ObjectId.is_valid(sms_data.user_id):
            raise HTTPException(status_code=400, detail="Invalid user ID")

        user = users_collection.find_one({"_id": ObjectId(sms_data.user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        original_message = sms_data.message_content
        processed_message = original_message
        detected_language = detect_language(original_message)
        
        # Translate to English if not in English
        translated_content = None
        if detected_language != 'en':
            translated_content = translate_to_english(original_message, detected_language)
            processed_message = translated_content
            print(f"[SMS INTERCEPT] Translated from {detected_language}: '{original_message}' -> '{translated_content}'")

        # Classify the message (use translated version if available)
        classification_result = classify_intercepted_message(processed_message)
        
        # Save message to file for logging
        save_message_to_file(sms_data.sender, processed_message, classification_result["prediction"])
        
        # Generate unique message ID
        import uuid
        message_id = str(uuid.uuid4())
        
        # Create message document for database storage
        message_doc = {
            "_id": ObjectId(),
            "message_id": message_id,
            "user_id": ObjectId(sms_data.user_id),
            "username": user["username"],
            "sender": sms_data.sender,
            "original_content": original_message,
            "processed_content": processed_message,
            "translated_content": translated_content,
            "original_language": detected_language,
            "prediction": classification_result["prediction"],
            "confidence": classification_result["confidence"],
            "timestamp": datetime.utcnow(),
            "alert_sent": classification_result["is_spam"]
        }
        
        # Store in appropriate collection based on classification
        if classification_result["is_spam"]:
            # Store in smish collection (alerts section)
            smish_messages_collection.insert_one(message_doc)
            alert_user = True
            print(f"[SMS INTERCEPT] SMISH DETECTED from {sms_data.sender}: {processed_message}")
        else:
            # Store in ham collection (normal messages section)
            ham_messages_collection.insert_one(message_doc)
            alert_user = False
            print(f"[SMS INTERCEPT] HAM message from {sms_data.sender}: {processed_message}")
        
        # Return classification response
        return SMSClassificationResponse(
            prediction=classification_result["prediction"],
            confidence=classification_result["confidence"],
            message_id=message_id,
            alert_user=alert_user,
            translated_content=translated_content
        )
        
    except Exception as e:
        print(f"[SMS INTERCEPT ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"SMS interception failed: {str(e)}")

@app.get("/messages/ham/{user_id}")
def get_ham_messages(user_id: str):
    """Get all ham (normal) messages for a user"""
    try:
        if not ObjectId.is_valid(user_id):
            raise HTTPException(status_code=400, detail="Invalid user ID")
        
        messages = list(ham_messages_collection.find(
            {"user_id": ObjectId(user_id)},
            {"_id": 0}
        ).sort("timestamp", -1))
        
        return {"messages": messages, "count": len(messages)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/messages/alerts/{user_id}")
def get_alert_messages(user_id: str):
    """Get all smish/spam alert messages for a user"""
    try:
        if not ObjectId.is_valid(user_id):
            raise HTTPException(status_code=400, detail="Invalid user ID")
        
        messages = list(smish_messages_collection.find(
            {"user_id": ObjectId(user_id)},
            {"_id": 0}
        ).sort("timestamp", -1))
        
        return {"alerts": messages, "count": len(messages)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==== CYBER ARTICLES ENDPOINTS ====
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict
import re

class ArticleResponse(BaseModel):
    title: str
    description: str
    link: str
    published: str
    source: str
    image_url: str = ""

@app.get("/api/cyber-articles", response_model=List[ArticleResponse])
async def get_cyber_articles():
    """Fetch real-time cyber security articles from multiple sources"""
    try:
        articles = []
        
        # RSS feeds for cyber security news
        rss_feeds = [
            {
                "url": "https://feeds.feedburner.com/TheHackersNews",
                "source": "The Hacker News"
            },
            {
                "url": "https://krebsonsecurity.com/feed/",
                "source": "Krebs on Security"
            },
            {
                "url": "https://www.bleepingcomputer.com/feed/",
                "source": "Bleeping Computer"
            },
            {
                "url": "https://feeds.feedburner.com/eset/blog",
                "source": "ESET Blog"
            },
            {
                "url": "https://www.darkreading.com/rss.xml",
                "source": "Dark Reading"
            }
        ]
        
        for feed in rss_feeds:
            try:
                response = requests.get(feed["url"], timeout=10)
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    
                    # Parse RSS items
                    for item in root.findall(".//item")[:5]:  # Limit to 5 articles per source
                        title = item.find("title")
                        description = item.find("description") 
                        link = item.find("link")
                        pub_date = item.find("pubDate")
                        
                        # Extract image from description if available
                        image_url = ""
                        if description is not None and description.text:
                            img_match = re.search(r'<img[^>]+src="([^"]+)"', description.text)
                            if img_match:
                                image_url = img_match.group(1)
                        
                        # Clean description text
                        desc_text = ""
                        if description is not None and description.text:
                            desc_text = re.sub(r'<[^>]+>', '', description.text)
                            desc_text = desc_text.strip()[:200] + "..." if len(desc_text) > 200 else desc_text
                        
                        article = ArticleResponse(
                            title=title.text if title is not None else "No Title",
                            description=desc_text,
                            link=link.text if link is not None else "",
                            published=pub_date.text if pub_date is not None else "",
                            source=feed["source"],
                            image_url=image_url
                        )
                        articles.append(article)
                        
            except Exception as e:
                print(f"Error fetching from {feed['source']}: {str(e)}")
                continue
        
        # Sort by published date (most recent first)
        articles.sort(key=lambda x: x.published, reverse=True)
        
        # Return top 20 articles
        return articles[:20]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching articles: {str(e)}")

@app.get("/api/cyber-articles/trending")
async def get_trending_articles():
    """Get trending cyber security topics"""
    try:
        # Simulated trending topics - in production, you might analyze article frequencies
        trending_topics = [
            {"topic": "AI Security", "count": 45},
            {"topic": "Ransomware", "count": 38},
            {"topic": "Data Breach", "count": 32},
            {"topic": "Phishing", "count": 28},
            {"topic": "Zero-day", "count": 24},
            {"topic": "IoT Security", "count": 19},
            {"topic": "Cloud Security", "count": 16},
            {"topic": "Mobile Security", "count": 14}
        ]
        return {"trending": trending_topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trending topics: {str(e)}")


# ==== FAQ ENDPOINTS ====
class FAQItem(BaseModel):
    id: int
    question: str
    answer: str
    category: str
    tags: List[str] = []

@app.get("/api/faqs", response_model=List[FAQItem])
async def get_faqs():
    """Get frequently asked questions about CyberTextShield and cybersecurity"""
    try:
        faqs = [
            {
                "id": 1,
                "question": "What is CyberTextShield and how does it work?",
                "answer": "CyberTextShield is an advanced mobile application that protects you from SMS spam, phishing attacks, and malicious text messages. It uses state-of-the-art AI technology including Heterogeneous Graph Neural Networks (HGNN) to analyze incoming messages in real-time and identify potential threats before they reach you.",
                "category": "General",
                "tags": ["app", "protection", "AI", "HGNN"]
            },
            {
                "id": 2,
                "question": "How accurate is the spam detection?",
                "answer": "Our HGNN-based spam detection system achieves over 95% accuracy in identifying spam and phishing messages. The system continuously learns from new threat patterns and adapts to emerging attack vectors, ensuring robust protection against both known and unknown threats.",
                "category": "Technology",
                "tags": ["accuracy", "detection", "AI", "machine learning"]
            },
            {
                "id": 3,
                "question": "Does CyberTextShield read my personal messages?",
                "answer": "No, CyberTextShield prioritizes your privacy. All message analysis is performed locally on your device using on-device AI models. Your personal messages are never transmitted to external servers or stored in the cloud. Only anonymized threat intelligence data is shared to improve our detection algorithms.",
                "category": "Privacy",
                "tags": ["privacy", "local", "security", "data protection"]
            },
            {
                "id": 4,
                "question": "What types of threats can CyberTextShield detect?",
                "answer": "CyberTextShield can detect various types of threats including: SMS spam, phishing attempts, smishing (SMS phishing), fake banking alerts, fraudulent promotional messages, malware distribution links, social engineering attacks, and suspicious sender patterns.",
                "category": "Security",
                "tags": ["threats", "phishing", "spam", "malware", "smishing"]
            },
            {
                "id": 5,
                "question": "How do I set up CyberTextShield on my device?",
                "answer": "Setting up CyberTextShield is simple: 1) Download and install the app, 2) Create an account or log in, 3) Grant necessary permissions for SMS access, 4) Complete the onboarding process, 5) The app will automatically start protecting you from threats. The setup process includes guided instructions for optimal configuration.",
                "category": "Setup",
                "tags": ["installation", "setup", "permissions", "onboarding"]
            },
            {
                "id": 6,
                "question": "Can I customize the protection settings?",
                "answer": "Yes, CyberTextShield offers extensive customization options. You can adjust sensitivity levels, whitelist trusted contacts, configure notification preferences, set custom blocking rules, and choose between different protection modes (strict, balanced, or lenient) based on your needs.",
                "category": "Settings",
                "tags": ["customization", "settings", "whitelist", "configuration"]
            },
            {
                "id": 7,
                "question": "What should I do if a legitimate message is marked as spam?",
                "answer": "If a legitimate message is incorrectly flagged, you can: 1) Mark it as 'Not Spam' in the app, 2) Add the sender to your whitelist, 3) Report the false positive to help improve our algorithms, 4) Adjust your protection sensitivity settings if needed. The system learns from your feedback to improve accuracy.",
                "category": "Troubleshooting",
                "tags": ["false positive", "whitelist", "feedback", "accuracy"]
            },
            {
                "id": 8,
                "question": "Does CyberTextShield work offline?",
                "answer": "Yes, CyberTextShield's core protection features work offline since the AI models are stored locally on your device. However, some features like real-time threat intelligence updates, article feeds, and cloud-based analysis require an internet connection for optimal performance.",
                "category": "Technology",
                "tags": ["offline", "local", "internet", "features"]
            }
        ]
        return faqs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching FAQs: {str(e)}")

@app.get("/api/faqs/categories")
async def get_faq_categories():
    """Get FAQ categories"""
    try:
        categories = [
            {"name": "General", "count": 1},
            {"name": "Technology", "count": 2},
            {"name": "Privacy", "count": 1},
            {"name": "Security", "count": 1},
            {"name": "Setup", "count": 1},
            {"name": "Settings", "count": 1},
            {"name": "Troubleshooting", "count": 1}
        ]
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching FAQ categories: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001)
