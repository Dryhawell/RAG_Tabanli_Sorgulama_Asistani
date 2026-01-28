# LLM Destekli PDF / Not Sorgulama Asistanı (RAG PoC)

## Proje Amacı
Kullanıcının yüklediği PDF veya TXT dokümanlarını analiz ederek, yalnızca bu doküman içeriklerine dayalı yanıtlar üreten bir RAG (Retrieval Augmented Generation) tabanlı asistan sunmak. Hallucination önlemek için LLM, sadece sağlanan bağlama dayanarak cevap verir ve metinde yoksa açıkça bildirir.

## Kullanılan Teknolojiler
- Python, Streamlit (UI)
- PDF okuma: PyMuPDF (pymupdf) ve pdfplumber (gerektiğinde)
- Embedding: sentence-transformers (`all-MiniLM-L6-v2`)
- Vektör Veritabanı: FAISS (CPU)
- LLM: OpenAI API veya lokal Ollama (LLaMA / Mistral / Phi)

## RAG Mimarisine Kısa Bakış
1. Doküman yüklenir (PDF/TXT)
2. Metin çıkarılır ve normalize edilir
3. Metin 500–800 kelimelik chunk’lara %10–15 overlap ile bölünür
4. Chunk’lar embedding vektörlerine çevrilir (MiniLM 384-dim)
5. Vektörler FAISS indeksinde saklanır (IndexFlatIP + ID eşlemesi)
6. Kullanıcı soru sorar; soru da embed edilir
7. En alakalı chunk’lar bulunur (top‑k)
8. Yalnızca bu chunk’lar ve katı kurallar LLM’e verilir
9. LLM sadece bağlamdan yanıt üretir; metinde yoksa “Bu bilgi dokümanda bulunmamaktadır” der

## Hallucination Önleme Kuralları
- "Sadece aşağıdaki metne dayanarak cevap ver"
- "Metinde yoksa ‘Bu bilgi dokümanda bulunmamaktadır’ de"
- Genel bilgi kullanımı yasaktır; cevapta alıntı ve chunk referansları (dosya, sayfa, chunk id) verilir

## Sınırlamalar
- OCR (tarama PDF’ler) kapsam dışıdır (PyMuPDF/pdfplumber metin olmayan sayfalarda sınırlı)
- Küçük yerel LLM modellerinde (Ollama, CPU) hız/kalite kısıtları olabilir
- İngilizce dışı metinlerde MiniLM performansı düşebilir (gelecekte çok dilli model düşünülebilir)

## Kurulum ve Çalıştırma (Windows, PowerShell)
1) Python 3.10/3.11 sanal ortam (önerilir):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Bağımlılıklar:

```powershell
pip install -r requirements.txt
```

3) LLM sağlayıcı seçimi:
- Ollama (varsayılan): Ollama'yı kurup çalıştırın. Uygulama, `http://localhost:11434` adresini kontrol eder.
- OpenAI: Ortam değişkenini ayarlayın ve bir model seçin (örn. `gpt-3.5-turbo`).

```powershell
$env:OPENAI_API_KEY = "<anahtarınız>"
```

4) Uygulamayı başlatma:

```powershell
streamlit run app/ui.py
```

5) Kullanım:
- Sol taraftan sağlayıcı ve model seçin.
- PDF/TXT dosyalarınızı yükleyin (data/ klasörüne kaydedilir, indeks oluşturulur).
- Sohbet kutusuna sorunuzu yazın. Altta ilgili chunk’lar ve skorlar gösterilir.

İpucu: “İndeksi Yeniden Oluştur” butonu, `data/` klasöründeki dosyalardan indeksi sıfırdan kurar.

## Dosya Yapısı
- `app/ui.py`: Streamlit arayüzü
- `rag/readers.py`: PDF/TXT okuma
- `rag/chunking.py`: chunking kuralları
- `rag/embed.py`: embedding üretimi
- `rag/index.py`: FAISS indeks ve kalıcılık
- `rag/llm.py`: LLM istemcileri (OpenAI/Ollama)
- `rag/prompt.py`: katı RAG prompt şablonu
- `rag/types.py`: metadata modelleri
- `indexes/`, `metadata/`, `data/`: kalıcı klasörler

## Testler
Basit smoke testleri `pytest` ile çalıştırabilirsiniz:

```powershell
pytest -q
```

Not: Testler model indirmesi yapmaz; FAISS ve chunking mantığını küçük örneklerle doğrular.
