## Install:
- n8n
- ollama

## Credential:
- Supabase
- Google Cloud API
- Postgres
- Zalobot

## Process:
### 1. Xây dựng kết nối với ZaloBot

Bước 1: Tạo Webhook (Ngrok for localhost)

Bước 2: Gửi Webhook url cho ZaloBot bằng HTTP Post

Bước 3: Thiết lập kết nối LLM - Agent

Bước 4: Tạo các node theo từng chức năng:
- SendMessage
- SendImage
- SendSticker ... (Xem thêm trên ZaloBot Docs)

### 2. Xây dựng kiến trúc RAG

Bước 1: Tạo và kết nối node dữ liệu (Docs, Excel, PDF-binary)

Bước 2: Tạo bảng trong Supabase

Bước 3: Tạo embedding từ dữ liệu và gửi vào bảng trong Supabase

Bước 4: Tạo node truy vấn dữ liệu từ Supabase

Xây dựng hoàn chỉnh: Gộp 1 và 2 lại với nhau

## Reference:
[1] https://bot.zapps.me/docs

[2] https://www.youtube.com/watch?v=fD2QRsMCpTY

[3] https://www.youtube.com/watch?v=uA9BIGT8hvw&t=136s
