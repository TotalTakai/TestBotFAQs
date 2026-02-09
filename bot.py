# bot.py
# Discord bot that answers questions when mentioned

import os
import discord
from dotenv import load_dotenv
import asyncio

from qa import answer_question

# ---------------- ENV ----------------
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# ---------------- DISCORD SETUP ----------------
intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)

# ---------------- EVENTS ----------------
@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Check if the bot was mentioned
    if bot.user in message.mentions:
        # Remove the mention from the message content
        question = message.content.replace(f"<@{bot.user.id}>", "").strip()

        if not question:
            await message.channel.send(
                "❓ Please ask a question after mentioning me."
            )
            return

        # Show typing indicator while processing
        async with message.channel.typing():
            try:
                # Run blocking QA in a separate thread
                answer = await asyncio.to_thread(answer_question, question)
                await message.channel.send(answer)
            except Exception as e:
                await message.channel.send("❌ Error while answering your question.")
                print("Error:", e)

# ---------------- RUN ----------------
bot.run(DISCORD_TOKEN)
