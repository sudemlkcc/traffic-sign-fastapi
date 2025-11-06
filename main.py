from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import os

app = FastAPI(
    title="Traffic Sign Classification API",
    description="Trafik işareti sınıflandırma servisi",
    version="1.0.0"
)

# Model yolu
MODEL_PATH = os.getenv("MODEL_PATH", "models/traffic_sign_model.h5")

# Trafik işareti etiketleri (GTSRB)
LABELS = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Model yükleme
model = None

@app.on_event("startup")
async def load_model():
    """Uygulama başlatıldığında modeli yükle"""
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✓ Model başarıyla yüklendi: {MODEL_PATH}")
    except Exception as e:
        print(f"✗ Model yüklenemedi: {str(e)}")
        raise


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Görüntüyü model için hazırlar
    
    Args:
        image_bytes: Görüntü byte verisi
        
    Returns:
        np.ndarray: İşlenmiş görüntü dizisi
    """
    # Bytes'tan görüntüye dönüştür
    image = Image.open(BytesIO(image_bytes))
    
    # RGB'ye dönüştür
    image = image.convert('RGB')
    
    # NumPy dizisine çevir
    image = np.array(image)
    
    # 32x32 boyutuna getir
    image = cv2.resize(image, (32, 32))
    
    # Normalize et
    image = image.astype("float32") / 255.0
    
    # Batch boyutu ekle
    image = np.expand_dims(image, axis=0)
    
    return image


@app.get("/")
async def root():
    """Ana endpoint - API bilgisi"""
    return {
        "message": "Traffic Sign Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API bilgisi",
            "/health": "Sağlık kontrolü",
            "/predict": "Trafik işareti tahmini (POST)",
            "/docs": "API dokümantasyonu"
        }
    }


@app.get("/health")
async def health_check():
    """Servis sağlık kontrolü"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model yüklenmedi")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": MODEL_PATH
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Trafik işareti tahmini yapar
    
    Args:
        file: Yüklenen görüntü dosyası
        
    Returns:
        JSON: Tahmin sonuçları
    """
    # Model kontrolü
    if model is None:
        raise HTTPException(status_code=503, detail="Model yüklenmedi")
    
    # Dosya tipi kontrolü
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Geçersiz dosya tipi. Lütfen bir görüntü dosyası yükleyin."
        )
    
    try:
        # Görüntüyü oku
        image_bytes = await file.read()
        
        # Görüntüyü işle
        processed_image = preprocess_image(image_bytes)
        
        # Tahmin yap
        predictions = model.predict(processed_image, verbose=0)
        
        # En yüksek olasılıklı sınıfı bul
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        # Sınıf etiketini al
        label = LABELS.get(predicted_class, f"Unknown Class {predicted_class}")
        
        # Top 3 tahmin
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                "class_id": int(idx),
                "label": LABELS.get(int(idx), f"Unknown Class {idx}"),
                "confidence": float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]
        
        # Sonuç
        result = {
            "success": True,
            "prediction": {
                "class_id": predicted_class,
                "label": label,
                "confidence": confidence
            },
            "top_3_predictions": top_3_predictions,
            "filename": file.filename
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin sırasında hata oluştu: {str(e)}"
        )


@app.get("/labels")
async def get_labels():
    """Tüm sınıf etiketlerini döndürür"""
    return {
        "total_classes": len(LABELS),
        "labels": LABELS
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7001)