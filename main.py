import discord
from discord.ext import commands, tasks
from discord import app_commands
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import io
import os
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Bot setup ─────────────────────────────────────────────────────────────────
TOKEN = os.environ.get("DISCORD_TOKEN", "")
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# ── In-memory storage ─────────────────────────────────────────────────────────
# { guild_id: { ticker: [ {type, period, color}, ... ] } }
ma_config: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

# [ {guild_id, channel_id, ticker, price, direction, triggered} ]
alerts: list[dict] = []

# Color palette for MA lines
MA_COLORS = [
    "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF",
    "#FF922B", "#CC5DE8", "#20C997", "#F06595",
    "#74C0FC", "#A9E34B",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def ticker_idx(raw: str) -> str:
    """Append .JK if not already present."""
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
    if pct > 0:
        return discord.Color.green()
    if pct < 0:
        return discord.Color.red()
    return discord.Color.greyple()


TIMEFRAME_MAP = {
    "1d":  ("1d",  "5m",  mdates.HourLocator(interval=1),   mdates.DateFormatter("%H:%M")),
    "5d":  ("5d",  "15m", mdates.HourLocator(interval=4),   mdates.DateFormatter("%d/%m %H:%M")),
    "1mo": ("1mo", "1h",  mdates.DayLocator(interval=3),    mdates.DateFormatter("%d/%m")),
    "3mo": ("3mo", "1d",  mdates.WeekdayLocator(byweekday=0), mdates.DateFormatter("%d/%m")),
    "6mo": ("6mo", "1d",  mdates.MonthLocator(),            mdates.DateFormatter("%b")),
    "1y":  ("1y",  "1d",  mdates.MonthLocator(),            mdates.DateFormatter("%b '%y")),
}


def calculate_ma(series: pd.Series, ma_type: str, period: int) -> pd.Series:
    if ma_type.upper() == "SMA":
        return series.rolling(window=period).mean()
    elif ma_type.upper() == "EMA":
        return series.ewm(span=period, adjust=False).mean()
    elif ma_type.upper() == "WMA":
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    return series.rolling(window=period).mean()


async def fetch_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    loop = asyncio.get_event_loop()
    try:
        df = await loop.run_in_executor(
            None,
            lambda: yf.download(ticker, period=period, interval=interval,
                                 auto_adjust=True, progress=False)
        )
        if df is None or df.empty:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception as e:
        log.error(f"fetch_stock_data error: {e}")
        return None


async def fetch_info(ticker: str) -> dict | None:
    loop = asyncio.get_event_loop()
    try:
        info = await loop.run_in_executor(None, lambda: yf.Ticker(ticker).info)
        return info
    except Exception as e:
        log.error(f"fetch_info error: {e}")
        return None


# ── Chart generator ───────────────────────────────────────────────────────────

def generate_chart(df: pd.DataFrame, ticker: str, timeframe: str,
                   ma_list: list[dict]) -> io.BytesIO:

    _, _, x_locator, x_formatter = TIMEFRAME_MAP[timeframe]

    # ── Layout ──
    fig = plt.figure(figsize=(14, 9), facecolor="#0D1117")
    ax_price  = fig.add_axes([0.07, 0.32, 0.91, 0.62])   # price panel
    ax_volume = fig.add_axes([0.07, 0.05, 0.91, 0.22], sharex=ax_price)  # volume panel

    for ax in (ax_price, ax_volume):
        ax.set_facecolor("#0D1117")
        ax.tick_params(colors="#C9D1D9", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")

    ax_price.grid(color="#21262D", linestyle="--", linewidth=0.5, alpha=0.7)
    ax_volume.grid(color="#21262D", linestyle="--", linewidth=0.5, alpha=0.4)

    # ── Candlesticks ──
    dates  = mdates.date2num(df.index.to_pydatetime())
    opens  = df["Open"].values.astype(float)
    closes = df["Close"].values.astype(float)
    highs  = df["High"].values.astype(float)
    lows   = df["Low"].values.astype(float)
    volumes = df["Volume"].values.astype(float)

    # candle width proportional to data density
    if len(dates) > 1:
        w = (dates[1] - dates[0]) * 0.7
    else:
        w = 0.0005

    for i in range(len(dates)):
        is_up = closes[i] >= opens[i]
        col   = "#26A641" if is_up else "#F85149"
        body_lo = min(opens[i], closes[i])
        body_hi = max(opens[i], closes[i])
        # body
        ax_price.add_patch(Rectangle(
            (dates[i] - w/2, body_lo), w, body_hi - body_lo,
            color=col, zorder=2
        ))
        # wick
        ax_price.plot(
            [dates[i], dates[i]], [lows[i], highs[i]],
            color=col, linewidth=0.8, zorder=1
        )
        # volume bar
        vcol = "#26A641" if is_up else "#F85149"
        ax_volume.bar(dates[i], volumes[i], width=w, color=vcol, alpha=0.7)

    # ── Moving Averages ──
    close_series = df["Close"].squeeze()
    for idx_ma, ma in enumerate(ma_list):
        ma_values = calculate_ma(close_series, ma["type"], ma["period"])
        color = MA_COLORS[idx_ma % len(MA_COLORS)]
        label = f"{ma['type']}{ma['period']}"
        ax_price.plot(
            dates, ma_values.values,
            color=color, linewidth=1.4, label=label, zorder=3
        )

    # ── Legend ──
    if ma_list:
        leg = ax_price.legend(
            loc="upper left", facecolor="#161B22",
            edgecolor="#30363D", labelcolor="#C9D1D9",
            fontsize=8, framealpha=0.9
        )

    # ── X-axis formatting ──
    ax_price.xaxis.set_major_locator(x_locator)
    ax_price.xaxis.set_major_formatter(x_formatter)
    ax_volume.xaxis.set_major_locator(x_locator)
    ax_volume.xaxis.set_major_formatter(x_formatter)
    plt.setp(ax_price.get_xticklabels(), visible=False)
    plt.setp(ax_volume.get_xticklabels(), rotation=30, ha="right")

    # ── Price annotation ──
    last_close = closes[-1]
    ax_price.axhline(last_close, color="#8B949E", linewidth=0.8, linestyle=":")
    ax_price.text(
        dates[-1], last_close,
        f"  {last_close:,.0f}",
        color="#C9D1D9", fontsize=8, va="center"
    )

    # ── Titles ──
    name = ticker.replace(".JK", "")
    pct  = ((closes[-1] - closes[0]) / closes[0]) * 100 if closes[0] != 0 else 0
    sign = "▲" if pct >= 0 else "▼"
    col_title = "#26A641" if pct >= 0 else "#F85149"

    ax_price.set_title(
        f"{name}  {sign} {abs(pct):.2f}%   [{timeframe}]",
        color=col_title, fontsize=13, fontweight="bold", pad=10
    )
    ax_price.set_ylabel("Price (IDR)", color="#8B949E", fontsize=8)
    ax_volume.set_ylabel("Volume", color="#8B949E", fontsize=8)

    ax_price.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax_volume.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: fmt_number(x))
    )

    plt.setp(ax_price.get_yticklabels(), color="#C9D1D9")
    plt.setp(ax_volume.get_yticklabels(), color="#8B949E")

    # ── Watermark ──
    fig.text(0.98, 0.01, "IDX Stock Bot", color="#30363D",
             fontsize=8, ha="right", va="bottom")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#0D1117")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Events ────────────────────────────────────────────────────────────────────

@bot.event
async def on_ready():
    log.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    try:
        synced = await bot.tree.sync()
        log.info(f"Synced {len(synced)} slash commands")
    except Exception as e:
        log.error(f"Sync error: {e}")
    check_alerts.start()


# ── Slash commands ─────────────────────────────────────────────────────────────

@bot.tree.command(name="price", description="Tampilkan harga saham IDX saat ini")
@app_commands.describe(ticker="Kode saham IDX, contoh: BBCA atau BBCA.JK")
async def price(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    t = ticker_idx(ticker)

    info = await fetch_info(t)
    if not info or "regularMarketPrice" not in info:
        # fallback with yf.download
        df = await fetch_stock_data(t, "2d", "1d")
        if df is None or df.empty:
            await interaction.followup.send(
                f"❌ Ticker **{t}** tidak ditemukan. Pastikan kode saham benar."
            )
            return
        close_vals = df["Close"].values.astype(float)
        current = float(close_vals[-1])
        prev    = float(close_vals[-2]) if len(close_vals) > 1 else current
        change  = current - prev
        pct     = (change / prev * 100) if prev else 0
        vol     = float(df["Volume"].values[-1])
        high    = float(df["High"].values[-1])
        low     = float(df["Low"].values[-1])
        name    = t
    else:
        current = info.get("regularMarketPrice") or info.get("currentPrice", 0)
        prev    = info.get("regularMarketPreviousClose") or info.get("previousClose", current)
        change  = current - prev
        pct     = (change / prev * 100) if prev else 0
        vol     = info.get("regularMarketVolume") or info.get("volume", 0)
        high    = info.get("regularMarketDayHigh") or info.get("dayHigh", 0)
        low     = info.get("regularMarketDayLow") or info.get("dayLow", 0)
        name    = info.get("longName") or info.get("shortName") or t

    arrow = "🟢 ▲" if change >= 0 else "🔴 ▼"
    embed = discord.Embed(
        title=f"📈 {t.replace('.JK','')} — {name}",
        color=color_for_change(pct),
        timestamp=datetime.utcnow()
    )
    embed.add_field(name="💰 Harga", value=f"**Rp {current:,.0f}**", inline=True)
    embed.add_field(name="📊 Perubahan",
                    value=f"{arrow} Rp {abs(change):,.0f} ({abs(pct):.2f}%)", inline=True)
    embed.add_field(name="\u200b", value="\u200b", inline=False)
    embed.add_field(name="🔼 High", value=f"Rp {high:,.0f}", inline=True)
    embed.add_field(name="🔽 Low",  value=f"Rp {low:,.0f}",  inline=True)
    embed.add_field(name="📦 Volume", value=fmt_number(vol), inline=True)
    embed.set_footer(text="Data via Yahoo Finance · IDX Stock Bot")
    await interaction.followup.send(embed=embed)


@bot.tree.command(name="chart", description="Generate chart candlestick saham IDX")
@app_commands.describe(
    ticker="Kode saham IDX, contoh: BBCA",
    timeframe="Pilih timeframe: 1d | 5d | 1mo | 3mo | 6mo | 1y"
)
@app_commands.choices(timeframe=[
    app_commands.Choice(name="1 Hari",   value="1d"),
    app_commands.Choice(name="5 Hari",   value="5d"),
    app_commands.Choice(name="1 Bulan",  value="1mo"),
    app_commands.Choice(name="3 Bulan",  value="3mo"),
    app_commands.Choice(name="6 Bulan",  value="6mo"),
    app_commands.Choice(name="1 Tahun",  value="1y"),
])
async def chart(interaction: discord.Interaction, ticker: str, timeframe: str = "1mo"):
    await interaction.response.defer()
    t = ticker_idx(ticker)

    period, interval, _, _ = TIMEFRAME_MAP[timeframe]
    df = await fetch_stock_data(t, period, interval)
    if df is None or df.empty:
        await interaction.followup.send(
            f"❌ Data untuk **{t}** tidak tersedia. Coba ticker lain."
        )
        return

    guild_id = interaction.guild_id or 0
    ma_list  = ma_config[guild_id][t.upper()]

    loop = asyncio.get_event_loop()
    buf  = await loop.run_in_executor(
        None, lambda: generate_chart(df, t, timeframe, ma_list)
    )

    # Build caption embed
    close_vals = df["Close"].values.astype(float)
    pct = ((close_vals[-1] - close_vals[0]) / close_vals[0] * 100) if close_vals[0] != 0 else 0
    embed = discord.Embed(
        title=f"📊 Chart {t.replace('.JK','')} [{timeframe}]",
        color=color_for_change(pct)
    )
    if ma_list:
        ma_text = "  |  ".join([f"{m['type']}{m['period']}" for m in ma_list])
        embed.add_field(name="📐 Moving Averages", value=ma_text, inline=False)
    else:
        embed.add_field(
            name="💡 Tip", 
            value="Tambah MA dengan `/addma` contoh: `/addma BBCA EMA 21`",
            inline=False
        )
    embed.set_footer(text="IDX Stock Bot · Data via Yahoo Finance")
    embed.set_image(url="attachment://chart.png")

    await interaction.followup.send(
        embed=embed,
        file=discord.File(buf, filename="chart.png")
    )


@bot.tree.command(name="addma", description="Tambahkan Moving Average ke chart ticker tertentu")
@app_commands.describe(
    ticker="Kode saham IDX, contoh: BBCA",
    ma_type="Tipe MA: SMA / EMA / WMA",
    period="Periode MA, contoh: 20"
)
@app_commands.choices(ma_type=[
    app_commands.Choice(name="SMA – Simple Moving Average",       value="SMA"),
    app_commands.Choice(name="EMA – Exponential Moving Average",  value="EMA"),
    app_commands.Choice(name="WMA – Weighted Moving Average",     value="WMA"),
])
async def addma(interaction: discord.Interaction, ticker: str, ma_type: str, period: int):
    if period < 2 or period > 500:
        await interaction.response.send_message(
            "❌ Period harus antara **2** dan **500**.", ephemeral=True
        )
        return

    t        = ticker_idx(ticker)
    guild_id = interaction.guild_id or 0
    ma_list  = ma_config[guild_id][t.upper()]

    # prevent duplicates
    for ma in ma_list:
        if ma["type"] == ma_type.upper() and ma["period"] == period:
            await interaction.response.send_message(
                f"⚠️ **{ma_type.upper()}{period}** sudah ada di {t}.", ephemeral=True
            )
            return

    if len(ma_list) >= 10:
        await interaction.response.send_message(
            "❌ Maksimal **10 MA** per ticker. Gunakan `/clearma` untuk reset.", ephemeral=True
        )
        return

    color_idx = len(ma_list)
    ma_config[guild_id][t.upper()].append({
        "type": ma_type.upper(),
        "period": period,
        "color": MA_COLORS[color_idx % len(MA_COLORS)]
    })

    current_mas = ma_config[guild_id][t.upper()]
    ma_text = "\n".join([
        f"`{MA_COLORS[i % len(MA_COLORS)]}` **{m['type']}{m['period']}**"
        for i, m in enumerate(current_mas)
    ])

    embed = discord.Embed(
        title=f"✅ MA Ditambahkan — {t.replace('.JK','')}",
        description=f"**{ma_type.upper()}{period}** berhasil ditambahkan!",
        color=discord.Color.green()
    )
    embed.add_field(
        name=f"📐 MA Aktif ({len(current_mas)}/10)",
        value=ma_text,
        inline=False
    )
    embed.set_footer(text="Gunakan /chart untuk melihat chart dengan MA")
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="clearma", description="Hapus semua MA yang sudah diset untuk ticker tertentu")
@app_commands.describe(ticker="Kode saham IDX, contoh: BBCA")
async def clearma(interaction: discord.Interaction, ticker: str):
    t        = ticker_idx(ticker)
    guild_id = interaction.guild_id or 0
    count    = len(ma_config[guild_id][t.upper()])
    ma_config[guild_id][t.upper()].clear()
    await interaction.response.send_message(
        f"🗑️ **{count} MA** untuk **{t.replace('.JK','')}** berhasil dihapus."
    )


@bot.tree.command(name="listma", description="Tampilkan daftar MA yang aktif untuk ticker tertentu")
@app_commands.describe(ticker="Kode saham IDX, contoh: BBCA")
async def listma(interaction: discord.Interaction, ticker: str):
    t        = ticker_idx(ticker)
    guild_id = interaction.guild_id or 0
    ma_list  = ma_config[guild_id][t.upper()]

    if not ma_list:
        await interaction.response.send_message(
            f"📭 Belum ada MA untuk **{t.replace('.JK','')}**. Gunakan `/addma` untuk menambahkan.",
            ephemeral=True
        )
        return

    embed = discord.Embed(
        title=f"📐 MA Aktif — {t.replace('.JK','')}",
        color=discord.Color.blurple()
    )
    lines = []
    for i, m in enumerate(ma_list):
        col = MA_COLORS[i % len(MA_COLORS)]
        lines.append(f"**{i+1}.** {m['type']}{m['period']}  `{col}`")
    embed.description = "\n".join(lines)
    embed.set_footer(text="Gunakan /clearma untuk reset semua MA")
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="alert", description="Set alert harga untuk saham IDX")
@app_commands.describe(
    ticker="Kode saham IDX, contoh: BBCA",
    harga="Target harga (IDR)",
    direction="above = harga naik ke target | below = harga turun ke target"
)
@app_commands.choices(direction=[
    app_commands.Choice(name="above – notif jika harga NAIK ke target", value="above"),
    app_commands.Choice(name="below – notif jika harga TURUN ke target", value="below"),
])
async def alert(interaction: discord.Interaction, ticker: str, harga: float, direction: str):
    t        = ticker_idx(ticker)
    guild_id = interaction.guild_id or 0

    # Max 5 alerts per user
    user_alerts = [a for a in alerts if a["user_id"] == interaction.user.id]
    if len(user_alerts) >= 5:
        await interaction.response.send_message(
            "❌ Maksimal **5 alert** aktif per user. Tunggu hingga ada yang terpicu.",
            ephemeral=True
        )
        return

    alerts.append({
        "guild_id":   guild_id,
        "channel_id": interaction.channel_id,
        "user_id":    interaction.user.id,
        "ticker":     t,
        "price":      harga,
        "direction":  direction,
        "triggered":  False
    })

    dir_text = "**NAIK** ke atas" if direction == "above" else "**TURUN** ke bawah"
    embed = discord.Embed(
        title="🔔 Alert Dipasang!",
        description=f"Kamu akan dinotifikasi ketika **{t.replace('.JK','')}** {dir_text} **Rp {harga:,.0f}**",
        color=discord.Color.yellow()
    )
    embed.add_field(name="Ticker",    value=t,                       inline=True)
    embed.add_field(name="Target",    value=f"Rp {harga:,.0f}",     inline=True)
    embed.add_field(name="Kondisi",   value=direction.upper(),        inline=True)
    embed.set_footer(text="Alert aktif selama bot nyala · Maks 5 alert per user")
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="myalerts", description="Tampilkan alert yang kamu pasang")
async def myalerts(interaction: discord.Interaction):
    user_alerts = [a for a in alerts
                   if a["user_id"] == interaction.user.id and not a["triggered"]]
    if not user_alerts:
        await interaction.response.send_message(
            "📭 Kamu belum punya alert aktif. Gunakan `/alert` untuk membuat.", ephemeral=True
        )
        return

    embed = discord.Embed(
        title="🔔 Alert Aktif Kamu",
        color=discord.Color.yellow()
    )
    for i, a in enumerate(user_alerts, 1):
        dir_emoji = "⬆️" if a["direction"] == "above" else "⬇️"
        embed.add_field(
            name=f"{i}. {a['ticker'].replace('.JK','')} {dir_emoji}",
            value=f"Target: **Rp {a['price']:,.0f}**",
            inline=True
        )
    await interaction.response.send_message(embed=embed, ephemeral=True)


@bot.tree.command(name="foreignflow", description="Tampilkan foreign net buy/sell saham IDX")
@app_commands.describe(ticker="Kode saham IDX (opsional), contoh: BBCA")
async def foreignflow(interaction: discord.Interaction, ticker: str = None):
    await interaction.response.defer()

    if ticker:
        # Single stock foreign flow (using institutional holders as proxy)
        t    = ticker_idx(ticker)
        loop = asyncio.get_event_loop()
        try:
            stock = yf.Ticker(t)
            inst  = await loop.run_in_executor(None, lambda: stock.institutional_holders)
            major = await loop.run_in_executor(None, lambda: stock.major_holders)
            info  = await loop.run_in_executor(None, lambda: stock.info)

            embed = discord.Embed(
                title=f"🌏 Foreign Flow — {t.replace('.JK','')}",
                description=f"**{info.get('longName', t)}**",
                color=discord.Color.blue(),
                timestamp=datetime.utcnow()
            )

            # Major holders summary
            if major is not None and not major.empty:
                try:
                    rows = major.iloc[:, :2].values.tolist()
                    holder_text = "\n".join([f"**{v[0]}** — {v[1]}" for v in rows[:4]])
                    embed.add_field(name="📊 Major Holders", value=holder_text, inline=False)
                except Exception:
                    pass

            # Top institutional holders
            if inst is not None and not inst.empty:
                inst_sorted = inst.sort_values("Value", ascending=False).head(5)
                lines = []
                for _, row in inst_sorted.iterrows():
                    name_h = str(row.get("Holder", row.get("Name", "N/A")))[:30]
                    val    = row.get("Value", 0)
                    pct    = row.get("% Out", row.get("pctHeld", 0))
                    if pct and pct < 1:
                        pct = pct * 100
                    lines.append(f"• **{name_h}** — {fmt_number(val)} ({pct:.2f}%)")
                embed.add_field(
                    name="🏦 Top 5 Institutional Holders",
                    value="\n".join(lines) or "Tidak tersedia",
                    inline=False
                )
            else:
                embed.add_field(
                    name="🏦 Institutional Holders",
                    value="Data tidak tersedia untuk ticker ini.",
                    inline=False
                )

            embed.add_field(
                name="⚠️ Catatan",
                value=(
                    "Data foreign net buy/sell real-time hanya tersedia via IDXE atau "
                    "broker API tertentu. Data ini menampilkan institutional holders sebagai proxy."
                ),
                inline=False
            )
            embed.set_footer(text="Data via Yahoo Finance · IDX Stock Bot")
            await interaction.followup.send(embed=embed)

        except Exception as e:
            log.error(f"foreignflow error: {e}")
            await interaction.followup.send(
                f"❌ Gagal mengambil data foreign flow untuk **{t}**."
            )

    else:
        # Generic market-wide info
        embed = discord.Embed(
            title="🌏 IDX Foreign Flow Overview",
            description=(
                "Data foreign net buy/sell real-time secara keseluruhan tidak tersedia "
                "via Yahoo Finance.\n\n"
                "**Sumber data resmi:**\n"
                "• 🔗 [IDX.co.id](https://www.idx.co.id/)\n"
                "• 🔗 [IDXE (IDX e-Statistic)](https://idxe.co.id/)\n"
                "• 🔗 [Stockbit.com](https://stockbit.com/)\n"
                "• 🔗 [RTI Business](https://rti.tech/)\n\n"
                "Gunakan `/foreignflow <ticker>` untuk data per saham."
            ),
            color=discord.Color.blue()
        )
        embed.set_footer(text="IDX Stock Bot")
        await interaction.followup.send(embed=embed)


@bot.tree.command(name="compare", description="Bandingkan harga beberapa saham sekaligus")
@app_commands.describe(tickers="Masukkan 2-5 ticker dipisah spasi, contoh: BBCA BBRI TLKM")
async def compare(interaction: discord.Interaction, tickers: str):
    await interaction.response.defer()
    ticker_list = [ticker_idx(t) for t in tickers.split()[:5]]
    if len(ticker_list) < 2:
        await interaction.followup.send("❌ Masukkan minimal **2 ticker**.")
        return

    embed = discord.Embed(
        title="📊 Perbandingan Saham IDX",
        color=discord.Color.blurple(),
        timestamp=datetime.utcnow()
    )

    for t in ticker_list:
        df = await fetch_stock_data(t, "2d", "1d")
        if df is None or df.empty:
            embed.add_field(name=t.replace(".JK",""), value="❌ Data tidak tersedia", inline=True)
            continue
        closes = df["Close"].values.astype(float)
        current = closes[-1]
        prev    = closes[-2] if len(closes) > 1 else closes[-1]
        pct     = (current - prev) / prev * 100 if prev else 0
        arrow   = "▲" if pct >= 0 else "▼"
        emoji   = "🟢" if pct >= 0 else "🔴"
        embed.add_field(
            name=f"{emoji} {t.replace('.JK','')}",
            value=f"**Rp {current:,.0f}**\n{arrow} {abs(pct):.2f}%",
            inline=True
        )

    embed.set_footer(text="Data via Yahoo Finance · IDX Stock Bot")
    await interaction.followup.send(embed=embed)


@bot.tree.command(name="help", description="Tampilkan semua perintah IDX Stock Bot")
async def help_cmd(interaction: discord.Interaction):
    embed = discord.Embed(
        title="📖 IDX Stock Bot — Panduan Lengkap",
        description="Bot saham IDX berbasis Yahoo Finance dengan fitur chart, alert, dan lebih!",
        color=discord.Color.blurple()
    )
    commands_info = [
        ("/price <ticker>",            "Harga terkini, perubahan %, volume, high/low"),
        ("/chart <ticker> <timeframe>","Chart candlestick + volume + MA  (1d/5d/1mo/3mo/6mo/1y)"),
        ("/addma <ticker> <type> <period>", "Tambah MA ke chart (SMA/EMA/WMA)  maks 10"),
        ("/listma <ticker>",           "Lihat daftar MA aktif untuk ticker"),
        ("/clearma <ticker>",          "Hapus semua MA untuk ticker"),
        ("/alert <ticker> <harga> <above/below>", "Set notifikasi harga (maks 5 alert)"),
        ("/myalerts",                  "Lihat alert aktif kamu"),
        ("/foreignflow [ticker]",      "Info institutional holders / foreign flow"),
        ("/compare <tickers>",         "Bandingkan 2–5 saham sekaligus"),
    ]
    for cmd, desc in commands_info:
        embed.add_field(name=f"`{cmd}`", value=desc, inline=False)

    embed.add_field(
        name="\n💡 Tips",
        value=(
            "• Semua ticker IDX otomatis ditambah `.JK`\n"
            "• Kamu bisa tambah banyak MA per ticker\n"
            "• Alert aktif sampai bot restart atau terpicu"
        ),
        inline=False
    )
    embed.set_footer(text="IDX Stock Bot · Powered by Yahoo Finance & discord.py")
    await interaction.response.send_message(embed=embed)


# ── Background task: Alert checker ────────────────────────────────────────────

@tasks.loop(minutes=2)
async def check_alerts():
    if not alerts:
        return

    pending = [a for a in alerts if not a["triggered"]]
    if not pending:
        return

    # Group tickers to minimize API calls
    unique_tickers = list({a["ticker"] for a in pending})
    prices: dict[str, float] = {}

    for t in unique_tickers:
        try:
            df = await fetch_stock_data(t, "1d", "1m")
            if df is not None and not df.empty:
                prices[t] = float(df["Close"].values[-1])
        except Exception:
            pass

    for a in pending:
        t       = a["ticker"]
        current = prices.get(t)
        if current is None:
            continue

        triggered = (
            (a["direction"] == "above" and current >= a["price"]) or
            (a["direction"] == "below" and current <= a["price"])
        )
        if triggered:
            a["triggered"] = True
            try:
                channel = bot.get_channel(a["channel_id"])
                if channel:
                    dir_emoji = "⬆️" if a["direction"] == "above" else "⬇️"
                    await channel.send(
                        f"🔔 <@{a['user_id']}> **Alert terpicu!**\n"
                        f"{dir_emoji} **{t.replace('.JK','')}** kini di **Rp {current:,.0f}** "
                        f"(target: Rp {a['price']:,.0f})"
                    )
            except Exception as e:
                log.error(f"Alert send error: {e}")


@check_alerts.before_loop
async def before_check():
    await bot.wait_until_ready()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not TOKEN:
        log.error("DISCORD_TOKEN environment variable not set!")
        exit(1)
    bot.run(TOKEN, log_handler=None)
