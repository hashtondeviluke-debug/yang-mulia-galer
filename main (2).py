import discord
from discord.ext import commands, tasks
from discord import app_commands
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import io
import os
import asyncio
from datetime import datetime
from collections import defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Thread pool khusus untuk operasi berat (chart, fetch)
executor = ThreadPoolExecutor(max_workers=4)

# ── Bot setup ─────────────────────────────────────────────────────────────────
TOKEN = os.environ.get("DISCORD_TOKEN", "")
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ── In-memory storage ─────────────────────────────────────────────────────────
ma_config: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
alerts: list[dict] = []

MA_COLORS = [
    "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF",
    "#FF922B", "#CC5DE8", "#20C997", "#F06595",
    "#74C0FC", "#A9E34B",
]

# Timeframe: candle_size -> (period_tarik, interval_yf)
TIMEFRAME_MAP = {
    "3m":  ("5d",  "3m"),
    "30m": ("1mo", "30m"),
    "1d":  ("6mo", "1d"),
    "1wk": ("2y",  "1wk"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def ticker_idx(raw: str) -> str:
    t = raw.upper().strip()
    return t if t.endswith(".JK") else f"{t}.JK"

def fmt_number(n) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    if abs(n) >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n:,.2f}"

def color_for_change(pct: float) -> discord.Color:
    if pct > 0:  return discord.Color.green()
    if pct < 0:  return discord.Color.red()
    return discord.Color.greyple()

def calculate_ma(series: pd.Series, ma_type: str, period: int) -> pd.Series:
    t = ma_type.upper()
    if t == "SMA":
        return series.rolling(window=period).mean()
    elif t == "EMA":
        return series.ewm(span=period, adjust=False).mean()
    elif t == "WMA":
        w = np.arange(1, period + 1)
        return series.rolling(window=period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
    return series.rolling(window=period).mean()

# ── Data fetching (sync, dijalankan di executor) ──────────────────────────────

def _fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False, timeout=8)
        if df is None or df.empty:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception as e:
        log.error(f"_fetch_ohlcv error: {e}")
        return None

def _fetch_info(ticker: str) -> dict | None:
    try:
        return yf.Ticker(ticker).info
    except Exception as e:
        log.error(f"_fetch_info error: {e}")
        return None

def _fetch_holders(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        return stock.info, stock.institutional_holders, stock.major_holders
    except Exception as e:
        log.error(f"_fetch_holders error: {e}")
        return {}, None, None

async def fetch_ohlcv(ticker, period, interval):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _fetch_ohlcv, ticker, period, interval)

async def fetch_info(ticker):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _fetch_info, ticker)

# ── Chart generator (sync, berjalan di executor) ──────────────────────────────

def _generate_chart(df: pd.DataFrame, ticker: str, timeframe: str, ma_list: list[dict]) -> io.BytesIO:
    fig = plt.figure(figsize=(12, 7), facecolor="#0D1117")
    ax_p = fig.add_axes([0.08, 0.30, 0.90, 0.64])
    ax_v = fig.add_axes([0.08, 0.05, 0.90, 0.20], sharex=ax_p)

    for ax in (ax_p, ax_v):
        ax.set_facecolor("#0D1117")
        ax.tick_params(colors="#C9D1D9", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363D")

    ax_p.grid(color="#21262D", linestyle="--", linewidth=0.4, alpha=0.6)
    ax_v.grid(color="#21262D", linestyle="--", linewidth=0.4, alpha=0.4)

    x     = np.arange(len(df))
    op    = df["Open"].values.astype(float)
    cl    = df["Close"].values.astype(float)
    hi    = df["High"].values.astype(float)
    lo    = df["Low"].values.astype(float)
    vol   = df["Volume"].values.astype(float)
    w     = 0.6

    for i in range(len(x)):
        up  = cl[i] >= op[i]
        col = "#26A641" if up else "#F85149"
        ax_p.add_patch(Rectangle(
            (x[i] - w/2, min(op[i], cl[i])), w, abs(cl[i] - op[i]),
            color=col, zorder=2
        ))
        ax_p.plot([x[i], x[i]], [lo[i], hi[i]], color=col, linewidth=0.7, zorder=1)
        ax_v.bar(x[i], vol[i], width=w, color=col, alpha=0.7)

    # Moving Averages
    cs = df["Close"].squeeze()
    for idx, ma in enumerate(ma_list):
        mv = calculate_ma(cs, ma["type"], ma["period"])
        ax_p.plot(x, mv.values,
                  color=MA_COLORS[idx % len(MA_COLORS)],
                  linewidth=1.3,
                  label=f"{ma['type']}{ma['period']}",
                  zorder=3)
    if ma_list:
        ax_p.legend(loc="upper left", facecolor="#161B22", edgecolor="#30363D",
                    labelcolor="#C9D1D9", fontsize=7)

    # Label X — ambil 5 tick saja biar ringan
    n_ticks = min(5, len(df))
    idxs    = np.linspace(0, len(df)-1, n_ticks, dtype=int)
    if timeframe == "3m":
        lbls = [df.index[i].strftime("%d/%m %H:%M") for i in idxs]
    elif timeframe == "30m":
        lbls = [df.index[i].strftime("%d/%m %H:%M") for i in idxs]
    elif timeframe == "1d":
        lbls = [df.index[i].strftime("%d %b '%y") for i in idxs]
    else:
        lbls = [df.index[i].strftime("%b '%y") for i in idxs]

    ax_p.set_xticks(idxs); ax_p.set_xticklabels([])
    ax_v.set_xticks(idxs); ax_v.set_xticklabels(lbls, rotation=25, ha="right", fontsize=7)

    # Harga terakhir
    lc = cl[-1]
    ax_p.axhline(lc, color="#8B949E", linewidth=0.7, linestyle=":")
    ax_p.text(x[-1], lc, f"  {lc:,.0f}", color="#C9D1D9", fontsize=7, va="center")

    name = ticker.replace(".JK", "")
    pct  = ((cl[-1] - cl[0]) / cl[0] * 100) if cl[0] != 0 else 0
    sign = "▲" if pct >= 0 else "▼"
    ax_p.set_title(f"{name}  {sign} {abs(pct):.2f}%  [{timeframe}]",
                   color="#26A641" if pct >= 0 else "#F85149",
                   fontsize=11, fontweight="bold", pad=8)
    ax_p.set_ylabel("Price (IDR)", color="#8B949E", fontsize=7)
    ax_v.set_ylabel("Vol", color="#8B949E", fontsize=7)
    ax_p.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_v.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_number(v)))
    plt.setp(ax_p.get_yticklabels(), color="#C9D1D9")
    plt.setp(ax_v.get_yticklabels(), color="#8B949E")
    fig.text(0.99, 0.01, "IDX Stock Bot", color="#30363D", fontsize=7, ha="right")

    buf = io.BytesIO()
    # dpi=100 lebih ringan dari 130
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0D1117")
    plt.close(fig)
    buf.seek(0)
    return buf

# ── Events ────────────────────────────────────────────────────────────────────

@bot.event
async def on_ready():
    log.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    # Paksa sync ulang semua guild + global
    await bot.tree.sync()
    log.info("Global slash commands synced!")
    for guild in bot.guilds:
        try:
            await bot.tree.sync(guild=guild)
            log.info(f"Synced to guild: {guild.name}")
        except Exception as e:
            log.warning(f"Guild sync failed {guild.name}: {e}")
    check_alerts.start()

# ── /price ────────────────────────────────────────────────────────────────────

@bot.tree.command(name="price", description="Tampilkan harga saham IDX saat ini")
@app_commands.describe(ticker="Kode saham IDX, contoh: BBCA atau BBCA.JK")
async def price(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    t    = ticker_idx(ticker)
    info = await fetch_info(t)

    if not info or "regularMarketPrice" not in info:
        df = await fetch_ohlcv(t, "2d", "1d")
        if df is None or df.empty:
            await interaction.followup.send(f"❌ Ticker **{t}** tidak ditemukan.")
            return
        cv = df["Close"].values.astype(float)
        current = float(cv[-1])
        prev    = float(cv[-2]) if len(cv) > 1 else current
        change  = current - prev
        pct     = change / prev * 100 if prev else 0
        vol     = float(df["Volume"].values[-1])
        high    = float(df["High"].values[-1])
        low     = float(df["Low"].values[-1])
        name    = t
    else:
        current = info.get("regularMarketPrice") or info.get("currentPrice", 0)
        prev    = info.get("regularMarketPreviousClose") or info.get("previousClose", current)
        change  = current - prev
        pct     = change / prev * 100 if prev else 0
        vol     = info.get("regularMarketVolume") or info.get("volume", 0)
        high    = info.get("regularMarketDayHigh") or info.get("dayHigh", 0)
        low     = info.get("regularMarketDayLow") or info.get("dayLow", 0)
        name    = info.get("longName") or info.get("shortName") or t

    arrow = "🟢 ▲" if change >= 0 else "🔴 ▼"
    embed = discord.Embed(
        title=f"📈 {t.replace('.JK','')} — {name}",
        color=color_for_change(pct), timestamp=datetime.utcnow()
    )
    embed.add_field(name="💰 Harga",     value=f"**Rp {current:,.0f}**", inline=True)
    embed.add_field(name="📊 Perubahan", value=f"{arrow} Rp {abs(change):,.0f} ({abs(pct):.2f}%)", inline=True)
    embed.add_field(name="\u200b",       value="\u200b", inline=False)
    embed.add_field(name="🔼 High",      value=f"Rp {high:,.0f}", inline=True)
    embed.add_field(name="🔽 Low",       value=f"Rp {low:,.0f}",  inline=True)
    embed.add_field(name="📦 Volume",    value=fmt_number(vol),    inline=True)
    embed.set_footer(text="Data via Yahoo Finance · IDX Stock Bot")
    await interaction.followup.send(embed=embed)

# ── /chart ────────────────────────────────────────────────────────────────────

@bot.tree.command(name="chart", description="Generate chart candlestick saham IDX")
@app_commands.describe(
    ticker="Kode saham IDX, contoh: BBCA",
    timeframe="Ukuran 1 candle"
)
@app_commands.choices(timeframe=[
    app_commands.Choice(name="3 Menit  (Intraday)",  value="3m"),
    app_commands.Choice(name="30 Menit (Intraday)",  value="30m"),
    app_commands.Choice(name="1 Hari   (Daily)",     value="1d"),
    app_commands.Choice(name="1 Minggu (Weekly)",    value="1wk"),
])
async def chart(interaction: discord.Interaction, ticker: str, timeframe: str = "1d"):
    # Langsung defer — kasih waktu lebih panjang
    await interaction.response.defer(thinking=True)
    t = ticker_idx(ticker)

    period, interval = TIMEFRAME_MAP[timeframe]
    df = await fetch_ohlcv(t, period, interval)
    if df is None or df.empty:
        await interaction.followup.send(
            f"❌ Data **{t}** tidak tersedia untuk timeframe `{timeframe}`.\n"
            f"(Yahoo Finance membatasi data intraday historis)"
        )
        return

    guild_id = interaction.guild_id or 0
    ma_list  = ma_config[guild_id][t.upper()]

    # Generate chart di thread pool agar tidak block event loop
    loop = asyncio.get_event_loop()
    buf  = await loop.run_in_executor(executor, _generate_chart, df, t, timeframe, ma_list)

    cv  = df["Close"].values.astype(float)
    pct = (cv[-1] - cv[0]) / cv[0] * 100 if cv[0] != 0 else 0
    embed = discord.Embed(
        title=f"📊 Chart {t.replace('.JK','')} [{timeframe}]",
        color=color_for_change(pct)
    )
    if ma_list:
        embed.add_field(name="📈 Moving Averages",
                        value="  |  ".join(f"{m['type']}{m['period']}" for m in ma_list),
                        inline=False)
    else:
        embed.add_field(name="💡 Tip",
                        value="Tambah MA: `/addma BBCA EMA 21`",
                        inline=False)
    embed.set_footer(text="IDX Stock Bot · Data via Yahoo Finance")
    embed.set_image(url="attachment://chart.png")
    await interaction.followup.send(embed=embed, file=discord.File(buf, filename="chart.png"))

# ── /addma ────────────────────────────────────────────────────────────────────

@bot.tree.command(name="addma", description="Tambahkan Moving Average ke chart ticker tertentu")
@app_commands.describe(
    ticker="Kode saham IDX, contoh: BBCA",
    ma_type="Tipe MA: SMA / EMA / WMA",
    period="Periode MA, contoh: 20"
)
@app_commands.choices(ma_type=[
    app_commands.Choice(name="SMA – Simple Moving Average",      value="SMA"),
    app_commands.Choice(name="EMA – Exponential Moving Average", value="EMA"),
    app_commands.Choice(name="WMA – Weighted Moving Average",    value="WMA"),
])
async def addma(interaction: discord.Interaction, ticker: str, ma_type: str, period: int):
    if period < 2 or period > 500:
        await interaction.response.send_message("❌ Period harus antara **2** dan **500**.", ephemeral=True)
        return

    t        = ticker_idx(ticker)
    guild_id = interaction.guild_id or 0
    ma_list  = ma_config[guild_id][t.upper()]

    for ma in ma_list:
        if ma["type"] == ma_type.upper() and ma["period"] == period:
            await interaction.response.send_message(
                f"⚠️ **{ma_type.upper()}{period}** sudah ada di {t}.", ephemeral=True)
            return

    if len(ma_list) >= 10:
        await interaction.response.send_message(
            "❌ Maksimal **10 MA** per ticker. Gunakan `/clearma` untuk reset.", ephemeral=True)
        return

    ma_config[guild_id][t.upper()].append({
        "type": ma_type.upper(), "period": period,
        "color": MA_COLORS[len(ma_list) % len(MA_COLORS)]
    })

    current_mas = ma_config[guild_id][t.upper()]
    lines = "\n".join(
        f"{i+1}. **{m['type']}{m['period']}**  `{MA_COLORS[i % len(MA_COLORS)]}`"
        for i, m in enumerate(current_mas)
    )
    embed = discord.Embed(
        title=f"✅ MA Ditambahkan — {t.replace('.JK','')}",
        description=f"**{ma_type.upper()}{period}** berhasil ditambahkan!",
        color=discord.Color.green()
    )
    embed.add_field(name=f"📐 MA Aktif ({len(current_mas)}/10)", value=lines, inline=False)
    embed.set_footer(text="Gunakan /chart untuk melihat chart dengan MA")
    await interaction.response.send_message(embed=embed)

# ── /clearma ──────────────────────────────────────────────────────────────────

@bot.tree.command(name="clearma", description="Hapus semua MA untuk ticker tertentu")
@app_commands.describe(ticker="Kode saham IDX, contoh: BBCA")
async def clearma(interaction: discord.Interaction, ticker: str):
    t        = ticker_idx(ticker)
    guild_id = interaction.guild_id or 0
    count    = len(ma_config[guild_id][t.upper()])
    ma_config[guild_id][t.upper()].clear()
    await interaction.response.send_message(
        f"🗑️ **{count} MA** untuk **{t.replace('.JK','')}** berhasil dihapus.")

# ── /listma ───────────────────────────────────────────────────────────────────

@bot.tree.command(name="listma", description="Tampilkan daftar MA aktif untuk ticker tertentu")
@app_commands.describe(ticker="Kode saham IDX, contoh: BBCA")
async def listma(interaction: discord.Interaction, ticker: str):
    t        = ticker_idx(ticker)
    guild_id = interaction.guild_id or 0
    ma_list  = ma_config[guild_id][t.upper()]

    if not ma_list:
        await interaction.response.send_message(
            f"📭 Belum ada MA untuk **{t.replace('.JK','')}**. Gunakan `/addma`.", ephemeral=True)
        return

    embed = discord.Embed(title=f"📐 MA Aktif — {t.replace('.JK','')}", color=discord.Color.blurple())
    embed.description = "\n".join(
        f"**{i+1}.** {m['type']}{m['period']}  `{MA_COLORS[i % len(MA_COLORS)]}`"
        for i, m in enumerate(ma_list)
    )
    embed.set_footer(text="Gunakan /clearma untuk reset")
    await interaction.response.send_message(embed=embed)

# ── /alert ────────────────────────────────────────────────────────────────────

@bot.tree.command(name="alert", description="Set alert harga untuk saham IDX")
@app_commands.describe(
    ticker="Kode saham IDX, contoh: BBCA",
    harga="Target harga (IDR)",
    direction="above = notif jika harga NAIK | below = notif jika harga TURUN"
)
@app_commands.choices(direction=[
    app_commands.Choice(name="above – notif jika harga NAIK ke target",  value="above"),
    app_commands.Choice(name="below – notif jika harga TURUN ke target", value="below"),
])
async def alert(interaction: discord.Interaction, ticker: str, harga: float, direction: str):
    t        = ticker_idx(ticker)
    guild_id = interaction.guild_id or 0
    active   = [a for a in alerts if a["user_id"] == interaction.user.id and not a["triggered"]]
    if len(active) >= 5:
        await interaction.response.send_message("❌ Maksimal **5 alert** aktif per user.", ephemeral=True)
        return

    alerts.append({
        "guild_id": guild_id, "channel_id": interaction.channel_id,
        "user_id": interaction.user.id, "ticker": t,
        "price": harga, "direction": direction, "triggered": False
    })
    dir_text = "**NAIK** ke atas" if direction == "above" else "**TURUN** ke bawah"
    embed = discord.Embed(
        title="🔔 Alert Dipasang!",
        description=f"Notif ketika **{t.replace('.JK','')}** {dir_text} **Rp {harga:,.0f}**",
        color=discord.Color.yellow()
    )
    embed.add_field(name="Ticker",  value=t,                   inline=True)
    embed.add_field(name="Target",  value=f"Rp {harga:,.0f}", inline=True)
    embed.add_field(name="Kondisi", value=direction.upper(),    inline=True)
    embed.set_footer(text="Maks 5 alert per user · Alert hilang saat bot restart")
    await interaction.response.send_message(embed=embed)

# ── /myalerts ─────────────────────────────────────────────────────────────────

@bot.tree.command(name="myalerts", description="Tampilkan alert aktif kamu")
async def myalerts(interaction: discord.Interaction):
    active = [a for a in alerts if a["user_id"] == interaction.user.id and not a["triggered"]]
    if not active:
        await interaction.response.send_message("📭 Belum ada alert aktif.", ephemeral=True)
        return
    embed = discord.Embed(title="🔔 Alert Aktif Kamu", color=discord.Color.yellow())
    for i, a in enumerate(active, 1):
        embed.add_field(
            name=f"{i}. {a['ticker'].replace('.JK','')} {'⬆️' if a['direction']=='above' else '⬇️'}",
            value=f"Target: **Rp {a['price']:,.0f}**", inline=True
        )
    await interaction.response.send_message(embed=embed, ephemeral=True)

# ── /foreignflow ──────────────────────────────────────────────────────────────

@bot.tree.command(name="foreignflow", description="Tampilkan foreign net buy/sell saham IDX")
@app_commands.describe(ticker="Kode saham IDX (opsional), contoh: BBCA")
async def foreignflow(interaction: discord.Interaction, ticker: str = None):
    await interaction.response.defer(thinking=True)
    if not ticker:
        embed = discord.Embed(
            title="🌏 IDX Foreign Flow Overview",
            description=(
                "Data market-wide tidak tersedia via Yahoo Finance.\n\n"
                "**Sumber data resmi:**\n"
                "• [IDX.co.id](https://www.idx.co.id/)\n"
                "• [IDXE](https://idxe.co.id/)\n"
                "• [Stockbit](https://stockbit.com/)\n"
                "• [RTI Business](https://rti.tech/)\n\n"
                "Gunakan `/foreignflow <ticker>` untuk data per saham."
            ),
            color=discord.Color.blue()
        )
        await interaction.followup.send(embed=embed)
        return

    t    = ticker_idx(ticker)
    loop = asyncio.get_event_loop()
    info, inst, major = await loop.run_in_executor(executor, _fetch_holders, t)

    embed = discord.Embed(
        title=f"🌏 Foreign Flow — {t.replace('.JK','')}",
        description=f"**{info.get('longName', t)}**",
        color=discord.Color.blue(), timestamp=datetime.utcnow()
    )
    if major is not None and not major.empty:
        try:
            rows = major.iloc[:, :2].values.tolist()
            embed.add_field(name="📊 Major Holders",
                            value="\n".join(f"**{v[0]}** — {v[1]}" for v in rows[:4]),
                            inline=False)
        except Exception:
            pass
    if inst is not None and not inst.empty:
        lines = []
        for _, row in inst.sort_values("Value", ascending=False).head(5).iterrows():
            nh  = str(row.get("Holder", row.get("Name", "N/A")))[:30]
            val = row.get("Value", 0)
            pct = row.get("% Out", row.get("pctHeld", 0))
            if pct and pct < 1: pct *= 100
            lines.append(f"• **{nh}** — {fmt_number(val)} ({pct:.2f}%)")
        embed.add_field(name="🏦 Top 5 Institutional Holders",
                        value="\n".join(lines) or "Tidak tersedia", inline=False)
    else:
        embed.add_field(name="🏦 Institutional Holders",
                        value="Data tidak tersedia untuk ticker ini.", inline=False)
    embed.add_field(name="⚠️ Catatan",
                    value="Data foreign real-time hanya tersedia via IDXE / broker API.",
                    inline=False)
    embed.set_footer(text="Data via Yahoo Finance · IDX Stock Bot")
    await interaction.followup.send(embed=embed)

# ── /compare ──────────────────────────────────────────────────────────────────

@bot.tree.command(name="compare", description="Bandingkan harga beberapa saham sekaligus")
@app_commands.describe(tickers="2-5 ticker dipisah spasi, contoh: BBCA BBRI TLKM")
async def compare(interaction: discord.Interaction, tickers: str):
    await interaction.response.defer(thinking=True)
    ticker_list = [ticker_idx(t) for t in tickers.split()[:5]]
    if len(ticker_list) < 2:
        await interaction.followup.send("❌ Masukkan minimal **2 ticker**.")
        return

    embed = discord.Embed(title="📊 Perbandingan Saham IDX",
                          color=discord.Color.blurple(), timestamp=datetime.utcnow())
    for t in ticker_list:
        df = await fetch_ohlcv(t, "2d", "1d")
        if df is None or df.empty:
            embed.add_field(name=t.replace(".JK",""), value="❌ Data tidak tersedia", inline=True)
            continue
        cv = df["Close"].values.astype(float)
        cur = cv[-1]; prv = cv[-2] if len(cv) > 1 else cv[-1]
        pct = (cur - prv) / prv * 100 if prv else 0
        embed.add_field(
            name=f"{'🟢' if pct>=0 else '🔴'} {t.replace('.JK','')}",
            value=f"**Rp {cur:,.0f}**\n{'▲' if pct>=0 else '▼'} {abs(pct):.2f}%",
            inline=True
        )
    embed.set_footer(text="Data via Yahoo Finance · IDX Stock Bot")
    await interaction.followup.send(embed=embed)

# ── /help ─────────────────────────────────────────────────────────────────────

@bot.tree.command(name="help", description="Tampilkan semua perintah IDX Stock Bot")
async def help_cmd(interaction: discord.Interaction):
    embed = discord.Embed(
        title="📖 IDX Stock Bot — Panduan",
        description="Bot saham IDX berbasis Yahoo Finance",
        color=discord.Color.blurple()
    )
    for cmd, desc in [
        ("/price <ticker>",                      "Harga terkini, change %, volume, high/low"),
        ("/chart <ticker> <timeframe>",          "Chart candlestick + volume + MA (3m/30m/1d/1wk)"),
        ("/addma <ticker> <type> <period>",      "Tambah MA ke chart (SMA/EMA/WMA) · maks 10"),
        ("/listma <ticker>",                     "Lihat daftar MA aktif"),
        ("/clearma <ticker>",                    "Hapus semua MA untuk ticker"),
        ("/alert <ticker> <harga> <above/below>","Set notifikasi harga · maks 5"),
        ("/myalerts",                            "Lihat alert aktif kamu"),
        ("/foreignflow [ticker]",                "Institutional holders / foreign flow"),
        ("/compare <tickers>",                   "Bandingkan 2–5 saham sekaligus"),
    ]:
        embed.add_field(name=f"`{cmd}`", value=desc, inline=False)
    embed.add_field(name="💡 Tips", value=(
        "• Ticker otomatis ditambah `.JK`\n"
        "• `/addma` bisa dipanggil berkali-kali\n"
        "• Alert hilang jika bot restart"
    ), inline=False)
    embed.set_footer(text="IDX Stock Bot · Yahoo Finance & discord.py")
    await interaction.response.send_message(embed=embed)

# ── Alert checker ─────────────────────────────────────────────────────────────

@tasks.loop(minutes=3)
async def check_alerts():
    pending = [a for a in alerts if not a["triggered"]]
    if not pending:
        return
    loop   = asyncio.get_event_loop()
    prices = {}
    for t in {a["ticker"] for a in pending}:
        df = await loop.run_in_executor(executor, _fetch_ohlcv, t, "1d", "5m")
        if df is not None and not df.empty:
            prices[t] = float(df["Close"].values[-1])

    for a in pending:
        cur = prices.get(a["ticker"])
        if cur is None: continue
        if (a["direction"] == "above" and cur >= a["price"]) or \
           (a["direction"] == "below" and cur <= a["price"]):
            a["triggered"] = True
            try:
                ch = bot.get_channel(a["channel_id"])
                if ch:
                    emoji = "⬆️" if a["direction"] == "above" else "⬇️"
                    await ch.send(
                        f"🔔 <@{a['user_id']}> **Alert terpicu!**\n"
                        f"{emoji} **{a['ticker'].replace('.JK','')}** kini "
                        f"**Rp {cur:,.0f}** (target: Rp {a['price']:,.0f})"
                    )
            except Exception as e:
                log.error(f"Alert send: {e}")

@check_alerts.before_loop
async def before_check():
    await bot.wait_until_ready()

# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not TOKEN:
        log.error("DISCORD_TOKEN not set!")
        exit(1)
    bot.run(TOKEN, log_handler=None)
