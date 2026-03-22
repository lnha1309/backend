import os
import io
import json
import base64
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import torch
from PIL import Image
import open_clip
from pymilvus import MilvusClient
from openai import AsyncOpenAI
import traceback 

# 1. TẢI BIẾN MÔI TRƯỜNG TỪ FILE .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Biến toàn cục (Global Variables)
model = None
preprocess = None
device = None
milvus_client = None
aclient = None
landmarks_db = {}

# 2. KHỞI TẠO TÀI NGUYÊN (Chỉ chạy 1 lần khi bật Server)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocess, device, milvus_client, aclient, landmarks_db
    print("⏳ Đang khởi tạo hệ thống AI Core...")

    # Init OpenAI client
    aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Init Zilliz (graceful — không crash nếu thiếu secret)
    try:
        milvus_client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
        print("✅ Zilliz connected")
    except Exception as e:
        print(f"⚠️  Zilliz init failed: {e}")

    # Load JSON data
    try:
        with open("hcm_landmarks_augmented.json", "r", encoding="utf-8") as f:
            landmarks_db = {item["id"]: item for item in json.load(f)}
        print(f"✅ Đã tải JSON: {len(landmarks_db)} địa danh.")
    except FileNotFoundError:
        print("⚠️  hcm_landmarks_augmented.json not found — chạy scripts/merge_data.py trước")
    except Exception as e:
        print(f"⚠️  Lỗi tải JSON: {e}")

    # Load CLIP Model (graceful — không crash nếu thiếu driver)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        model.eval().to(device)
        print(f"✅ CLIP model loaded trên {device.upper()}")
    except Exception as e:
        print(f"⚠️  CLIP model init failed: {e}")
        device = "cpu"

    yield

    print("🛑 Đang tắt Server...")

app = FastAPI(title="VietStory Lens API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "landmarks_count": len(landmarks_db),
        "device": device or "not_initialized",
    }

# 3. HÀM GỌI ELEVENLABS
async def generate_audio_base64(text: str) -> str:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data, headers=headers, timeout=20.0)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8')
        else:
            raise Exception(f"ElevenLabs Error: {response.text}")

# 4. API ENDPOINT (GIAO TIẾP VỚI FRONTEND)
@app.post("/api/v1/tour-guide")
async def get_tour_guide_data(file: UploadFile = File(...)):
    try:
        # BƯỚC 1: Xử lý ảnh & Gọi Zilliz
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        query_vector = image_features.cpu().numpy().tolist()[0]
        
        search_results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=1,
            output_fields=["landmark_id"] 
        )
        
        if not search_results or len(search_results[0]) == 0:
            raise HTTPException(status_code=404, detail="Không nhận diện được ảnh.")
            
        best_match = search_results[0][0]
        landmark_id = best_match["entity"]["landmark_id"]
        confidence = best_match["distance"]
        
        if confidence < 0.6:
            raise HTTPException(status_code=400, detail="Độ tin cậy quá thấp, vui lòng chụp lại.")

        # BƯỚC 2: Rút trích lịch sử từ JSON
        if landmark_id not in landmarks_db:
            raise HTTPException(status_code=500, detail="Không tìm thấy thông tin trong Database JSON.")
            
        info = landmarks_db[landmark_id]
        
        # BƯỚC 3: Sinh nội dung bằng OpenAI
        system_prompt = f"""
        You are an engaging and professional tour guide for the VietStory Lens application.
        The tourist is currently standing in front of: {info['name']}.
        Historical context: '{info['context_docs'][0]['content']}'.
        Visual elements the tourist is seeing right now: {", ".join(info['vision_tags'])}.
        
        INSTRUCTIONS:
        1. Tell a captivating and immersive story seamlessly blending the historical context and the visual elements.
        2. Keep the length strictly between 100-120 words.
        3. Your final response MUST be entirely in English.
        4. Do NOT use any markdown formatting (such as **, *, or #) because this text will be processed by a Text-to-Speech engine.
        """
        
        openai_res = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": "Tell me the story of this place."}],
            temperature=0.7
        )
        story_text = openai_res.choices[0].message.content.strip()

        # BƯỚC 4: Tạo Audio Base64 bằng ElevenLabs
        audio_base64 = await generate_audio_base64(story_text)

        # BƯỚC 5: Gói gọn trả về Frontend
        return {
            "status": "success",
            "data": {
                "landmark_id": landmark_id,
                "landmark_name": info["name"],
                "confidence_score": round(confidence, 4),
                "story_text": story_text,
                "audio_base64": audio_base64
            }
        }
        
    except HTTPException as http_error:
        # Nếu là lỗi do mình chủ động ném ra (400, 404), thì giữ nguyên ném thẳng về App
        raise http_error
        
    except Exception as e:
        # Nếu là lỗi sập hệ thống không lường trước (500)
        error_details = traceback.format_exc() # In ra tường tận từ dòng code nào
        print(f"🔥 LỖI SERVER CHI TIẾT:\n{error_details}") # Dòng này sẽ in thẳng lên Log của Google
        
        # Trả về cho Swagger/App biết đích danh tên Lỗi là gì (VD: KeyError, TypeError...)
        raise HTTPException(status_code=500, detail=f"Lỗi gốc: {type(e).__name__} - {str(e)}")
