# UITBOT: Production Cloud Deployment Guide

Hướng dẫn từng bước triển khai hệ sinh thái **UITBOT** lên máy chủ đám mây (Cloud VPS - Ubuntu 22.04 LTS / 24.04 LTS) với Docker Compose, Nginx Reverse Proxy, chứng chỉ bảo mật SSL Let's Encrypt và chính sách sao lưu tự động.

---

## 🖥️ 1. Yêu Cầu Cấu Hình Máy Chủ (System Requirements)

- **Hệ điều hành**: Ubuntu 22.04 LTS hoặc 24.04 LTS (x86_64).
- **CPU**: Tối thiểu 2 vCPU (khuyến nghị 4 vCPU để xử lý nhanh bước Reranking trên CPU).
- **RAM**: Tối thiểu 4GB RAM + 4GB Swap (Khuyến nghị 8GB RAM).
- **Dung lượng đĩa (Disk)**: Tối thiểu 20GB SSD khả dụng (cho Docker images, PostgreSQL volume và cache HuggingFace).
- **Cổng mạng (Ports)**:
  - `80/tcp` (HTTP) & `443/tcp` (HTTPS).
  - `22/tcp` (SSH).
  - *Lưu ý*: Cổng cơ sở dữ liệu `5433` KHÔNG mở ra internet công cộng (chỉ lắng nghe nội bộ).

---

## 🚀 2. Chuẩn Bị Môi Trường Trên Máy Chủ

Đăng nhập vào VPS qua SSH và cập nhật hệ điều hành:

```bash
# 1. Cập nhật hệ thống
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y curl git ufw fail2ban certbot python3-certbot-nginx

# 2. Cấu hình tường lửa UFW an toàn
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# 3. Cài đặt Docker & Docker Compose v2 (Official Script)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Kích hoạt nhóm docker ngay lập tức:
newgrp docker
docker --version
docker compose version
```

---

## 📦 3. Triển Khai UITBOT Bằng Docker Compose

```bash
# 1. Clone repository về máy chủ
cd /opt
sudo git clone https://github.com/huynhhoangtuvan11092006/UITBOT.git uitbot
sudo chown -R $USER:$USER /opt/uitbot
cd /opt/uitbot

# 2. Tạo file cấu hình môi trường sản xuất (.env)
cp .env.example .env
nano .env
```

### Thiết lập các biến môi trường quan trọng trong `.env`:
```ini
# Đổi mật khẩu database an toàn cho production
POSTGRES_PASSWORD=your_super_secret_production_password_2026

# Cổng LLM Gateway chính thức
OPENAI_API_KEY=sk-your-official-openai-or-gateway-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# Cấu hình máy chủ
DEBUG=false
HOST=0.0.0.0
PORT=8000
```

### Khởi động container & Nạp dữ liệu:
```bash
# 3. Build và khởi chạy các dịch vụ ở chế độ nền (Detached)
docker compose up -d --build

# 4. Kiểm tra trạng thái sức khỏe
docker compose ps
curl http://localhost:8000/health

# 5. Chạy pipeline nạp và số hóa toàn bộ quy chế vào pgvector
docker compose exec -T app python scripts/ingest.py
```

---

## 🔒 4. Cấu Hình Nginx Reverse Proxy & Chứng Chỉ SSL HTTPS

Tạo file cấu hình virtual host cho Nginx:

```bash
sudo nano /etc/nginx/sites-available/uitbot.conf
```

Dán nội dung cấu hình sau (thay `uitbot.yourdomain.com` bằng tên miền thực tế của bạn):

```nginx
server {
    server_name uitbot.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;

        # Hỗ trợ Server-Sent Events (SSE) Streaming
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;

        # Headers chuẩn xác thực
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Thời gian chờ cho các truy vấn suy luận LLM / Reranker
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }

    # Giới hạn kích thước tải lên
    client_max_body_size 25M;
}
```

Kích hoạt cấu hình và cấp phát chứng chỉ SSL tự động qua Let's Encrypt:

```bash
# Kích hoạt site
sudo ln -s /etc/nginx/sites-available/uitbot.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Cấp chứng chỉ SSL HTTPS miễn phí tự động gia hạn
sudo certbot --nginx -d uitbot.yourdomain.com
```

---

## 🔄 5. Sao Lưu & Phục Hồi Dữ Liệu Tự Động (Backup & Restore)

### Tạo script tự động sao lưu PostgreSQL pgvector:
Tạo file `/opt/uitbot/scripts/backup_db.sh`:

```bash
#!/usr/bin/env bash
set -e

BACKUP_DIR="/opt/uitbot/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$BACKUP_DIR"

echo "Đang tạo bản sao lưu cơ sở dữ liệu UITBOT tại: $TIMESTAMP..."
docker compose exec -T db pg_dump -U uitbot_user -d uitbot -F c > "$BACKUP_DIR/uitbot_backup_$TIMESTAMP.dump"

# Xóa các bản sao lưu cũ hơn 14 ngày để tiết kiệm dung lượng
find "$BACKUP_DIR" -type f -name "*.dump" -mtime +14 -delete
echo "Sao lưu hoàn tất thành công!"
```

Cấp quyền thực thi và tạo lịch chạy tự động bằng cron (hàng ngày lúc 02:00 sáng):

```bash
chmod +x /opt/uitbot/scripts/backup_db.sh
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/uitbot/scripts/backup_db.sh > /dev/null 2>&1") | crontab -
```

### Phục hồi cơ sở dữ liệu khi có sự cố:
```bash
docker compose exec -T db pg_restore -U uitbot_user -d uitbot --clean < /opt/uitbot/backups/uitbot_backup_YYYYMMDD_HHMMSS.dump
```

---

## 📈 6. Giám Sát & Quản Trị Hệ Thống

- **Xem logs trực tiếp theo thời gian thực**:
  ```bash
  docker compose logs -f --tail=100 app
  ```
- **Khởi động lại dịch vụ nhanh**:
  ```bash
  docker compose restart app
  ```
- **Kiểm tra tài nguyên tiêu thụ**:
  ```bash
  docker stats
  ```
- **Kiểm tra endpoint giám sát sức khỏe từ bên ngoài**:
  ```bash
  curl -s https://uitbot.yourdomain.com/health | jq .
  ```
