import discord
from discord.ext import commands
from discord import app_commands
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import io, os, asyncio

TOKEN = os.environ.get("DISCORD_TOKEN", "")
bot = commands.Bot(command_prefix="!", intents=discord.Intents.default())

@bot.event
async def on_ready():
    await bot.tree.sync()
    print("Bot is Online!")

@bot.tree.command(name="price", description="Cek harga saham")
async def price(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    t = ticker.upper() + ".JK" if not ticker.upper().endswith(".JK") else ticker.upper()
    stock = yf.Ticker(t)
    hist = stock.history(period="1d")
    if hist.empty: return await interaction.followup.send("Ticker tidak ditemukan.")
    price = hist['Close'].iloc[-1]
    await interaction.followup.send(f"Harga **{t}**: Rp {price:,.0f}")

@bot.tree.command(name="chart", description="Chart TradingView style (3m, 30m, 1d, 1wk)")
async def chart(interaction: discord.Interaction, ticker: str, timeframe: str = "1d"):
    await interaction.response.defer()
    t = ticker.upper() + ".JK" if not ticker.upper().endswith(".JK") else ticker.upper()
    
    # Map timeframe ke interval yfinance
    mapping = {"3m": ("5d", "3m"), "30m": ("1mo", "30m"), "1d": ("6mo", "1d"), "1wk": ("2y", "1wk")}
    period, interval = mapping.get(timeframe, ("6mo", "1d"))
    
    df = yf.download(t, period=period, interval=interval, progress=False)
    if df.empty: return await interaction.followup.send("Data gagal diambil.")
    
    buf = io.BytesIO()
    mpf.plot(df, type='candle', style='charles', savefig=buf, volume=True)
    buf.seek(0)
    await interaction.followup.send(file=discord.File(buf, filename="chart.png"))

@bot.tree.command(name="sharechart", description="Kirim hasil analisa TradingView kamu")
async def sharechart(interaction: discord.Interaction, ticker: str, link: str, note: str = ""):
    embed = discord.Embed(title=f"Analisa {ticker.upper()}", description=f"Catatan: {note}", color=discord.Color.blue())
    embed.add_field(name="Link Chart", value=link)
    await interaction.response.send_message(embed=embed)

bot.run(TOKEN)
