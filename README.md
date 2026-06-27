# 📈 IDX Stock Bot — Discord Bot untuk Saham IDX

Bot Discord untuk memantau saham IDX Indonesia dengan fitur chart candlestick,
moving average, price alert, dan foreign flow.

---

## ✨ Fitur

| Command | Deskripsi |
|---------|-----------|
| `/price <ticker>` | Harga terkini, change %, volume, high/low |
| `/chart <ticker> <timeframe>` | Candlestick chart + volume bar + MA |
| `/addma <ticker> <type> <period>` | Tambah Moving Average (SMA/EMA/WMA) ke chart |
| `/listma <ticker>` | Lihat semua MA aktif per ticker |
| `/clearma <ticker>` | Hapus semua MA untuk ticker tertentu |
| `/alert <ticker> <harga> <above/below>` | Set notifikasi harga |
| `/myalerts` | Lihat alert aktif kamu |
| `/foreignflow [ticker]` | Institutional holders / foreign flow info |
| `/compare <tickers>` | Bandingkan 2–5 saham sekaligus |
| `/help` | Tampilkan semua perintah |

---

## 🚀 Setup Lokal

### 1. Clone & Install

```bash
git clone <repo-url>
cd idx-stock-bot
pip install -r requirements.txt
```

### 2. Buat Discord Bot

1. Buka [Discord Developer Portal](https://discord.com/developers/applications)
2. Klik **New Application** → beri nama
3. Masuk ke tab **Bot** → klik **Add Bot**
4. Di bagian **Privileged Gateway Intents**, aktifkan:
   - ✅ Message Content Intent
5. Klik **Reset Token** → copy token-nya
6. Masuk ke tab **OAuth2 → URL Generator**:
   - Scope: `bot` + `applications.commands`
   - Bot Permissions: `Send Messages`, `Embed Links`, `Attach Files`, `Read Message History`
7. Copy URL dan buka di browser → invite bot ke server kamu

### 3. Jalankan Bot

```bash
export DISCORD_TOKEN="token_kamu_disini"
python main.py
```

---

## ☁️ Deploy ke Railway.app (24/7)

### Langkah-langkah:

**1. Push ke GitHub**
```bash
git init
git add .
git commit -m "IDX Stock Bot"
git remote add origin https://github.com/username/idx-stock-bot.git
git push -u origin main
```

**2. Buat Project di Railway**
1. Buka [railway.app](https://railway.app) → Login dengan GitHub
2. Klik **New Project** → **Deploy from GitHub repo**
3. Pilih repository kamu

**3. Set Environment Variable**
1. Di Railway, masuk ke project → tab **Variables**
2. Klik **New Variable**:
   - Key: `DISCORD_TOKEN`
   - Value: `token_discord_bot_kamu`
3. Klik **Add**

**4. Deploy**
- Railway akan otomatis detect `Procfile` dan deploy
- Cek tab **Deployments** untuk melihat logs
- Status ✅ = bot sudah nyala 24/7!

### Tips Railway:
- Free tier = $5 credit/bulan (cukup untuk bot ringan)
- Aktifkan **Restart on failure** di settings
- Gunakan tab **Logs** untuk debug

---

## 📊 Contoh Penggunaan

```
/price BBCA          → Harga BCA terkini
/price BBRI.JK       → Support format dengan .JK juga

/chart TLKM 1mo      → Chart 1 bulan TLKM
/chart BBCA 3mo      → Chart 3 bulan BCA

/addma BBCA EMA 21   → Tambah EMA 21 ke chart BBCA
/addma BBCA SMA 50   → Tambah SMA 50 ke chart BBCA
/addma BBCA EMA 200  → Tambah EMA 200 ke chart BBCA
/chart BBCA 6mo      → Chart akan menampilkan semua 3 MA di atas

/alert BBCA 10000 above  → Notif jika BBCA naik ke 10.000
/alert GOTO 50 below     → Notif jika GOTO turun ke 50

/compare BBCA BBRI BMRI TLKM  → Bandingkan 4 saham
/foreignflow BBCA             → Institutional holders BCA
```

---

## 📁 Struktur File

```
idx-stock-bot/
├── main.py           ← Kode utama bot
├── requirements.txt  ← Library yang dibutuhkan
├── Procfile          ← Konfigurasi process untuk Railway
├── railway.toml      ← Konfigurasi Railway
└── README.md         ← Panduan ini
```

---

## ⚠️ Catatan Penting

- **Data:** Semua data saham menggunakan Yahoo Finance (gratis, ada delay ~15 menit)
- **Alert:** Disimpan di memory — akan hilang jika bot restart
- **Foreign Flow:** Data real-time IDX hanya tersedia via API premium (Stockbit, RTI)
- **MA:** Maksimal 10 MA per ticker, tersimpan selama bot nyala
- **Ticker IDX:** Otomatis ditambah `.JK` jika belum ada

---

## 🔧 Troubleshooting

**Slash commands tidak muncul?**
- Tunggu 1–5 menit setelah bot join server
- Pastikan bot punya permission `applications.commands`

**Error `DISCORD_TOKEN not set`?**
- Set environment variable dulu sebelum jalankan
- Di Railway: tambahkan di tab Variables

**Chart blank / error?**
- Ticker mungkin tidak terdaftar di Yahoo Finance
- Coba format: `BBCA` atau `BBCA.JK`

**Data tidak update?**
- Yahoo Finance memiliki delay ~15 menit untuk IDX
- Untuk intraday (1d), gunakan data 1m atau 5m interval
