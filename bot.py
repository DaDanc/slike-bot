import os
import io
import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message, BufferedInputFile
from huggingface_hub import InferenceClient

# Tokene drži u environment varijablama, NE u kodu (inače ih jednom pushaš na GitHub i gotovo)
BOT_TOKEN = os.environ["BOT_TOKEN"]
HF_TOKEN = os.environ["HF_TOKEN"]

# Brz i kvalitetan model; možeš zamijeniti npr. sa "stabilityai/stable-diffusion-xl-base-1.0"
MODEL = "black-forest-labs/FLUX.1-schnell"

# Osnovni filter promptova (EN + HR). Nije savršen, ali blokira očite stvari.
BLOCKED_WORDS = {
    "nude", "naked", "nsfw", "sex", "sexy", "porn", "uncensored", "topless",
    "explicit", "hentai", "boobs", "tits", "dick", "pussy", "lingerie",
    "child", "kid", "teen", "loli", "minor",
    "gol", "gola", "golo", "seks", "porno", "sise", "dijete", "djeca", "maloljet",
}

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
hf = InferenceClient(provider="auto", api_key=HF_TOKEN)

# Da jedan user ne može spamati 20 zahtjeva odjednom
busy_users: set[int] = set()


def is_blocked(prompt: str) -> bool:
    text = prompt.lower()
    return any(word in text for word in BLOCKED_WORDS)


def generate(prompt: str) -> bytes:
    # text_to_image vraća PIL Image -> pretvaramo u PNG bytes
    image = hf.text_to_image(prompt, model=MODEL)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Pošalji mi opis slike, npr.\n"
        "'minimalistički logo za kafić, plava i bijela'\n"
        "'mačka astronaut na Mjesecu, digitalna umjetnost'"
    )


@dp.message(F.text)
async def handle_prompt(message: Message):
    user_id = message.from_user.id
    prompt = message.text.strip()

    if len(prompt) < 3:
        await message.answer("Prekratak opis, napiši malo više.")
        return

    if is_blocked(prompt):
        await message.answer("Taj prompt nije dozvoljen.")
        return

    if user_id in busy_users:
        await message.answer("Još generiram tvoju prošlu sliku, pričekaj.")
        return

    busy_users.add(user_id)
    status = await message.answer("Generiram... ⏳")
    try:
        # HF klijent je sinkroni, pa ga pokrećemo u threadu da ne blokira bota
        image_bytes = await asyncio.to_thread(generate, prompt)
        photo = BufferedInputFile(image_bytes, filename="image.png")
        await message.answer_photo(photo, caption=prompt[:1000])
    except Exception as e:
        logging.exception("Greška pri generiranju")
        await message.answer(f"Nešto je puklo: {e}")
    finally:
        busy_users.discard(user_id)
        await status.delete()


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
