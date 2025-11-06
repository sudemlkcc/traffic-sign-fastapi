# Traffic Sign Classification API 🚦

FastAPI ve Docker kullanarak trafik işareti sınıflandırma servisi.

## Kullanım

1. Model dosyasını `models/` klasörüne koyun
2. Docker container'ı başlatın:
```bash
docker-compose up --build
```
3. API'ye erişin: http://localhost:7001

## Endpoints

- `GET /` - API bilgisi
- `GET /health` - Sağlık kontrolü
- `POST /predict` - Tahmin (görüntü yükle)
- `GET /labels` - Tüm sınıf etiketleri
- `GET /docs` - API dokümantasyonu