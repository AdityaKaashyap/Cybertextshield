from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bson import ObjectId
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pickle
import numpy as np
from db_config import users_collection, permissions_collection

# === Load model and preprocessing ===
with open("vectorizer_fixed.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
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
model.load_state_dict(torch.load("hgnn_spam_classifier.pt", map_location=device))
model.eval()

# === FastAPI setup ===
app = FastAPI()

# ==== SCHEMAS ====

class RegisterRequest(BaseModel):
    username: str
    country_code:str
    phone_number: str

class PermissionsRequest(BaseModel):
    user_id: str
    allow_contacts: bool
    allow_location: bool
    allow_media: bool
    allow_call_logs: bool
    allow_sms: bool

class Message(BaseModel):
    text: str
    user_id: str

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

# === ROUTES ===

@app.post("/register")
def register_user(data: RegisterRequest):
    # Basic format check (you can add stricter regex or use phonenumbers lib)
    if not data.country_code.startswith("+") or not data.phone_number.isdigit():
        raise HTTPException(status_code=400, detail="Invalid phone number or country code format")

    full_phone = f"{data.country_code}{data.phone_number}"

    user_data = {
        "username": data.username,
        "country_code": data.country_code,
        "phone_number": full_phone
    }
    result = users_collection.insert_one(user_data)
    return {
        "user_id": str(result.inserted_id),
        "message": f"User {data.username} registered successfully with phone {full_phone}"
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

    vec = vectorizer.transform([msg.text]).toarray()
    x_all, edge_index = build_hypergraph_single(vec)

    with torch.no_grad():
        out = model(x_all, edge_index)
        num_features = vectorizer.max_features
        out_sample = out[num_features:]
        probs = F.softmax(out_sample, dim=1).squeeze().numpy()
        pred = np.argmax(probs)
        label = label_enc.inverse_transform([pred])[0]
        return {
            "user": user["username"],
            "message": msg.text,
            "prediction": label,
            "confidence": float(probs[pred]),
            "probabilities": {
                label_enc.inverse_transform([0])[0]: float(probs[0]),
                label_enc.inverse_transform([1])[0]: float(probs[1]),
            }
        }
