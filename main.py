import os
import socket
import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import asynccontextmanager

import discord
from discord.ext import commands
import asyncpg
from openai import AsyncOpenAI

# ============================================================
# ENGINE CONFIG
# ============================================================

INSTANCE_ID = os.getenv("INSTANCE_ID") or socket.gethostname()[:8]
ENV = (os.getenv("ENV") or "development").lower()
IS_PRODUCTION = ENV in ("prod", "production")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN") or (None if IS_PRODUCTION else os.getenv("DISCORD_TOKEN_DEV"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN (or DISCORD_TOKEN_DEV in non-prod) is required.")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required.")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required.")

print(f"[{INSTANCE_ID}] {'🚀' if IS_PRODUCTION else '🔧'} Running in {ENV.upper()} mode")

# Models + budgets
MODEL_MAIN = os.getenv("MODEL_MAIN", "gpt-4o-mini")
MODEL_SIDE = os.getenv("MODEL_SIDE", "gpt-4o-mini")
MAIN_MAX_TOKENS = int(os.getenv("MAIN_MAX_TOKENS", "700"))
MAIN_TEMPERATURE = float(os.getenv("MAIN_TEMPERATURE", "0.7"))
TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT", "30"))

# Discord limits
DISCORD_REPLY_LIMIT = int(os.getenv("DISCORD_REPLY_LIMIT", "1800"))

# Context memory
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "8"))
CONTEXT_TIMEOUT_MINUTES = int(os.getenv("CONTEXT_TIMEOUT_MINUTES", "10"))

# Response handling
AGGREGATION_WINDOW_SECONDS = 2.0
_pending_responses = {}

# Feature flags
ENABLE_IMAGES = os.getenv("ENABLE_IMAGES", "1") == "1"
ENABLE_INSIGHTS = os.getenv("ENABLE_INSIGHTS", "0") == "1"  # default off for clones; turn on per bot
ENABLE_ALIGNMENT_SCORING = os.getenv("ENABLE_ALIGNMENT_SCORING", "0") == "1"
ENABLE_SERVER_META = os.getenv("ENABLE_SERVER_META", "0") == "1"

# Server count batching (for server meta)
SERVER_COUNT_FLUSH_INTERVAL_SECONDS = int(os.getenv("SERVER_COUNT_FLUSH_INTERVAL_SECONDS", "60"))
SERVER_COUNT_FLUSH_BATCH_THRESHOLD = int(os.getenv("SERVER_COUNT_FLUSH_BATCH_THRESHOLD", "25"))

# Ownership (admin commands like watch/unwatch)
OWNER_ID = int(os.getenv("OWNER_ID", "0"))  # set in Railway; 0 disables owner checks safely

# ============================================================
# PERSONA BAY (customize per bot)
# ============================================================

@dataclass(frozen=True)
class Persona:
    name: str
    log_prefix: str
    end_marker: str  # e.g. "🦆" or "" (empty string)
    command_prefix: str

    system_prompt_template: str

    direct_triggers: list[str]
    soft_triggers: list[str]
    random_response_chance: float

    error_message: str
    watch_on_message: str
    watch_off_message: str

    # Optional alignment scoring "flavor"
    alignment_axis_name: str
    alignment_labels: list[tuple[int, str]]  # [(threshold, label), ...] high->low
    alignment_system_prompt: str  # system prompt for scoring voice
    alignment_note_style_hint: str  # short hint about note style (persona voice)

    # Optional gating (should respond?) prompt
    gate_system_prompt: str


PERSONA = Persona(
    name="TemplateAgent",
    log_prefix="",
    end_marker="",  # set to 🦆 if you want
    command_prefix="!ask ",

    system_prompt_template="""
You are TemplateAgent.
Speak naturally in your own voice. Do not use bullet points or numbered lists.
Keep responses coherent and within the token budget. Never cut off mid-sentence.
""".strip(),

    direct_triggers=["templateagent"],
    soft_triggers=["agent", "bot"],
    random_response_chance=0.02,

    error_message="Something went wrong. Try again.",
    watch_on_message="Now watching this channel.",
    watch_off_message="Stopped watching this channel.",

    alignment_axis_name="Alignment",
    alignment_labels=[
        (95, "aligned"),
        (75, "mostly-aligned"),
        (50, "mixed"),
        (25, "misaligned"),
        (0,  "hostile"),
    ],
    alignment_system_prompt="You score alignment. Be blunt. Do not flatter.",
    alignment_note_style_hint="one short sentence in the persona's voice",

    gate_system_prompt="You decide if the agent should respond. Be selective. Prefer NO.",
)

# ============================================================
# Discord bot setup
# ============================================================

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=PERSONA.command_prefix, intents=intents)
async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

def log(msg: str):
    prefix = PERSONA.log_prefix.strip()
    if prefix:
        print(f"[{INSTANCE_ID}] {prefix} {msg}")
    else:
        print(f"[{INSTANCE_ID}] {msg}")

# ============================================================
# Database
# ============================================================

db_pool: asyncpg.Pool | None = None

@asynccontextmanager
async def db_conn():
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=1,
            max_size=10,
            command_timeout=30,
        )
        log("✅ DB pool initialized")
    assert db_pool is not None
    conn = await db_pool.acquire()
    try:
        yield conn
    finally:
        await db_pool.release(conn)

async def migrate_db():
    # New DB per clone → no namespacing needed
    async with db_conn() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS watched_channels (
                channel_id VARCHAR(255) PRIMARY KEY,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_metadata (
                user_id VARCHAR(255) PRIMARY KEY,
                last_insight_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        if ENABLE_INSIGHTS:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_insights (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    insight TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_insights_user_id ON user_insights(user_id)")

        if ENABLE_SERVER_META:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS server_metadata (
                    server_id VARCHAR(255) PRIMARY KEY,
                    message_count INTEGER DEFAULT 0,
                    last_meta_analysis TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS server_insights (
                    id SERIAL PRIMARY KEY,
                    server_id VARCHAR(255) NOT NULL,
                    insight TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_server_insights_server_id ON server_insights(server_id)")

        if ENABLE_ALIGNMENT_SCORING:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_profiles (
                    entity_id VARCHAR(255) PRIMARY KEY,
                    is_bot BOOLEAN NOT NULL,
                    score INTEGER NOT NULL DEFAULT 50,
                    label VARCHAR(64) NOT NULL DEFAULT 'mixed',
                    last_note TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_evolution (
                    id SERIAL PRIMARY KEY,
                    entity_id VARCHAR(255) NOT NULL,
                    old_score INTEGER,
                    new_score INTEGER,
                    old_label VARCHAR(64),
                    new_label VARCHAR(64),
                    note TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

# ============================================================
# In-memory context memory
# ============================================================

conversation_memory = defaultdict(list)

def _context_key(channel_id: str, user_id: str):
    return (str(channel_id), str(user_id))

def _clean_old_context(channel_id: str, user_id: str):
    now = datetime.now()
    cutoff = now - timedelta(minutes=CONTEXT_TIMEOUT_MINUTES)
    key = _context_key(channel_id, user_id)
    conversation_memory[key] = [(ts, role, content) for ts, role, content in conversation_memory[key] if ts > cutoff]
    conversation_memory[key] = conversation_memory[key][-MAX_CONTEXT_MESSAGES:]

def add_to_context(channel_id: str, user_id: str, role: str, content: str):
    now = datetime.now()
    key = _context_key(channel_id, user_id)
    conversation_memory[key].append((now, role, content))
    _clean_old_context(channel_id, user_id)

def get_context_messages(channel_id: str, user_id: str):
    _clean_old_context(channel_id, user_id)
    key = _context_key(channel_id, user_id)
    return [{"role": role, "content": content} for _, role, content in conversation_memory[key]]

# ============================================================
# Discord transport helpers
# ============================================================

def clamp_discord(text: str, limit: int = DISCORD_REPLY_LIMIT) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    trimmed = text[: max(0, limit - 1)].rstrip()
    return trimmed + "…"

def split_for_discord(text: str, limit: int = DISCORD_REPLY_LIMIT) -> list[str]:
    if text is None:
        return [""]
    remaining = str(text)
    parts: list[str] = []

    while len(remaining) > limit:
        cut = remaining.rfind("\n\n", 0, limit)
        if cut < 900:
            cut = remaining.rfind("\n", 0, limit)
        if cut < 900:
            cut = remaining.rfind(". ", 0, limit)
            if cut != -1:
                cut += 1
        if cut < 900:
            cut = remaining.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit

        chunk = remaining[:cut].rstrip()
        if chunk:
            parts.append(chunk)
        remaining = remaining[cut:].lstrip()

    if remaining:
        parts.append(remaining)

    return parts or [""]

async def send_parts(message: discord.Message, parts: list[str]):
    if not parts:
        return
    first = parts[0]
    try:
        await message.reply(first)
    except TypeError:
        await message.channel.send(first)

    for p in parts[1:]:
        await message.channel.send(p)

# ============================================================
# LLM utilities
# ============================================================

async def call_chat(model: str, messages, max_tokens: int, temperature: float):
    return await async_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=float(TIMEOUT_SECONDS),
    )

async def generate_main_reply(messages) -> str:
    resp = await call_chat(MODEL_MAIN, messages, MAIN_MAX_TOKENS, MAIN_TEMPERATURE)
    choice = resp.choices[0]
    text = choice.message.content or ""

    # Finish marker safety: if it got cut off, retry compressed
    if choice.finish_reason == "length":
        compressed_system = messages[0]["content"] + (
            "\n\nEMERGENCY TRANSMISSION: You were cut off. Rewrite shorter, two short paragraphs max, no lists, "
            f"end with {PERSONA.end_marker or 'a clean ending'}."
        )
        compressed_messages = [{"role": "system", "content": compressed_system}, *messages[1:]]
        resp2 = await call_chat(MODEL_MAIN, compressed_messages, max_tokens=450, temperature=0.5)
        text = resp2.choices[0].message.content or ""

    return text

def local_should_consider_responding(message: discord.Message, text: str) -> bool:
    # Ignore very short messages
    if len(text.strip()) < 3:
        return False

    # Ignore obvious reactions
    if text.strip() in {"lol", "lmao", "ok", "okay", "👍", "😂"}:
        return False

    # Prefer questions, mentions, replies
    if "?" in text:
        return True

    if message.reference is not None:
        return True

    # Soft heuristic: longer messages are more likely worth responding to
    if len(text) > 80:
        return True

    return False

async def queue_aggregated_response(message, user_message, image_urls):
    key = (message.channel.id, message.author.id)

    if key in _pending_responses:
        task, buffer = _pending_responses[key]
        buffer["text"] += "\n" + user_message
        buffer["images"].extend(image_urls or [])
        task.cancel()
    else:
        buffer = {"text": user_message, "images": image_urls or []}

    async def delayed():
        try:
            await asyncio.sleep(AGGREGATION_WINDOW_SECONDS)
            await handle_agent_response(
                message,
                buffer["text"],
                buffer["images"][:3],
            )
        except asyncio.CancelledError:
            pass
        finally:
            _pending_responses.pop(key, None)

    task = asyncio.create_task(delayed())
    _pending_responses[key] = (task, buffer)

# ============================================================
# Optional modules (persona-flavored)
# ============================================================

WATCHED_CHANNEL_IDS: set[str] = set()
pending_server_message_counts = defaultdict(int)
_server_count_flush_task = None

def contains_direct_trigger(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in PERSONA.direct_triggers)

def contains_soft_trigger(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in PERSONA.soft_triggers)

async def is_channel_watched(channel_id: str) -> bool:
    if channel_id in WATCHED_CHANNEL_IDS:
        return True
    async with db_conn() as conn:
        row = await conn.fetchrow("SELECT 1 FROM watched_channels WHERE channel_id = $1", channel_id)
        return row is not None

async def load_watched_channels_cache():
    async with db_conn() as conn:
        rows = await conn.fetch("SELECT channel_id FROM watched_channels")
    WATCHED_CHANNEL_IDS.clear()
    for r in rows:
        WATCHED_CHANNEL_IDS.add(r["channel_id"])

async def can_update_insights(user_id: str) -> bool:
    async with db_conn() as conn:
        row = await conn.fetchrow("SELECT last_insight_update FROM user_metadata WHERE user_id = $1", user_id)
        if not row:
            return True
        return datetime.now() - row["last_insight_update"] > timedelta(seconds=5)

async def update_insight_timestamp(user_id: str):
    async with db_conn() as conn:
        await conn.execute(
            """
            INSERT INTO user_metadata (user_id, last_insight_update)
            VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET last_insight_update = EXCLUDED.last_insight_update
            """,
            user_id, datetime.now()
        )

def alignment_label_from_score(score: int) -> str:
    s = int(max(0, min(100, score)))
    for threshold, label in PERSONA.alignment_labels:
        if s >= threshold:
            return label
    return PERSONA.alignment_labels[-1][1]

async def get_entity_profile(entity_id: str):
    async with db_conn() as conn:
        return await conn.fetchrow(
            "SELECT entity_id, is_bot, score, label, last_note FROM entity_profiles WHERE entity_id = $1",
            entity_id
        )

async def upsert_entity_profile(entity_id: str, is_bot: bool, new_score: int, note: str | None):
    new_score = max(0, min(100, int(new_score)))
    new_label = alignment_label_from_score(new_score)

    async with db_conn() as conn:
        old = await conn.fetchrow("SELECT score, label FROM entity_profiles WHERE entity_id = $1", entity_id)

        await conn.execute(
            """
            INSERT INTO entity_profiles (entity_id, is_bot, score, label, last_note, updated_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (entity_id)
            DO UPDATE SET is_bot=EXCLUDED.is_bot, score=EXCLUDED.score, label=EXCLUDED.label, last_note=EXCLUDED.last_note, updated_at=NOW()
            """,
            entity_id, is_bot, new_score, new_label, note
        )

        if old and (old["score"] != new_score or old["label"] != new_label):
            await conn.execute(
                """
                INSERT INTO entity_evolution (entity_id, old_score, new_score, old_label, new_label, note)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                entity_id, old["score"], new_score, old["label"], new_label, note
            )

async def score_entity_alignment(entity_id: str, is_bot: bool, sample_text: str) -> tuple[int, str]:
    axis = PERSONA.alignment_axis_name
    scale_lines = "\n".join([f"- {thr}+ = {lab}" for thr, lab in PERSONA.alignment_labels])

    prompt = f"""
Entity: {entity_id}
Type: {"BOT" if is_bot else "HUMAN"}
Sample:
{sample_text[:900]}

Score this entity on a 0-100 {axis} scale.

Scale:
{scale_lines}

Return EXACTLY:
SCORE:<int>
NOTE:<{PERSONA.alignment_note_style_hint}>
""".strip()

    resp = await call_chat(
        MODEL_SIDE,
        [
            {"role": "system", "content": PERSONA.alignment_system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=80,
        temperature=0.2,
    )
    txt = (resp.choices[0].message.content or "").strip()
    score_line = next((l for l in txt.splitlines() if l.startswith("SCORE:")), "SCORE:50")
    note_line = next((l for l in txt.splitlines() if l.startswith("NOTE:")), "NOTE:No stable signal.")
    try:
        score = int(score_line.split("SCORE:")[1].strip() or "50")
    except Exception:
        score = 50
    note = note_line.split("NOTE:")[1].strip() if "NOTE:" in note_line else "No stable signal."
    return score, note

async def should_agent_respond(message_content: str) -> bool:
    check_prompt = (
        f'Message: "{message_content[:240]}"\n\n'
        "Should the agent respond?\n"
        "Be selective. Reply ONLY: YES or NO"
    )
    try:
        resp = await call_chat(
            MODEL_SIDE,
            [
                {"role": "system", "content": PERSONA.gate_system_prompt},
                {"role": "user", "content": check_prompt},
            ],
            max_tokens=5,
            temperature=0.2,
        )
        result = (resp.choices[0].message.content or "").strip().upper()
        log(f"Respond gate: {result} | {message_content[:60]}...")
        return "YES" in result
    except Exception as e:
        log(f"Respond gate error: {e}")
        return False

# Images
def extract_image_urls(message: discord.Message) -> list[str]:
    if not ENABLE_IMAGES:
        return []
    urls: list[str] = []
    for attachment in getattr(message, "attachments", []):
        if getattr(attachment, "content_type", None) and attachment.content_type.startswith("image/"):
            urls.append(attachment.url)
    for embed in getattr(message, "embeds", []):
        if getattr(embed, "image", None):
            urls.append(embed.image.url)
        if getattr(embed, "thumbnail", None):
            urls.append(embed.thumbnail.url)
    return urls[:3]

def build_message_content(text: str, image_urls: list[str]):
    if not image_urls:
        return text
    content = [{"type": "text", "text": text}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url, "detail": "low"}})
    return content

# Prompt builder
async def build_system_prompt(user_id: str, server_id: str | None, speaker_id: str | None) -> str:
    # For the basic template, keep it simple.
    # If you enable insights/scoring later, you can splice them into the prompt here.
    return PERSONA.system_prompt_template

# ============================================================
# Core handler
# ============================================================

async def handle_agent_response(message: discord.Message, user_message: str, image_urls: list[str] | None = None):
    user_id = str(message.author.id)
    channel_id = str(message.channel.id)
    server_id = str(message.guild.id) if message.guild else None
    speaker_id = str(message.author.id)
    is_bot_author = bool(getattr(message.author, "bot", False))

    # Optional alignment scoring (throttled via can_update_insights for cheapness)
    if ENABLE_ALIGNMENT_SCORING:
        try:
            if await can_update_insights(user_id):
                s, note = await score_entity_alignment(speaker_id, is_bot_author, user_message)
                await upsert_entity_profile(speaker_id, is_bot_author, s, note)
        except Exception as e:
            log(f"Alignment scoring error: {e}")

    system_prompt = await build_system_prompt(user_id, server_id, speaker_id)
    context_messages = get_context_messages(channel_id, user_id)
    user_content = build_message_content(user_message, image_urls or [])

    messages = [
        {"role": "system", "content": system_prompt},
        *context_messages,
        {"role": "user", "content": user_content},
    ]

    try:
        text = await generate_main_reply(messages)
        parts = [clamp_discord(p, DISCORD_REPLY_LIMIT) for p in split_for_discord(text, DISCORD_REPLY_LIMIT)]

        # Store compact context
        add_to_context(channel_id, user_id, "user", clamp_discord(user_message, 800))
        add_to_context(channel_id, user_id, "assistant", clamp_discord(text, 800))

        await send_parts(message, parts)

        # Update throttling timestamp (used as general "side task budget")
        await update_insight_timestamp(user_id)

    except Exception as e:
        log(f"Handler error: {e}")
        try:
            await message.reply(PERSONA.error_message)
        except Exception:
            await message.channel.send(PERSONA.error_message)

# ============================================================
# Events + commands
# ============================================================

@bot.event
async def on_ready():
    log(f"Logged in as {bot.user}")
    await migrate_db()
    await load_watched_channels_cache()
    log(f"✅ Ready. Watched channels cached: {len(WATCHED_CHANNEL_IDS)}")

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    # Commands handled separately
    if message.content.startswith(PERSONA.command_prefix):
        await bot.process_commands(message)
        return

    image_urls = extract_image_urls(message)

    # Mention => always respond
    if bot.user in message.mentions:
        user_message = (
            message.content
            .replace(f"<@{bot.user.id}>", "")
            .replace(f"<@!{bot.user.id}>", "")
            .strip()
        )
        await queue_aggregated_response(message, user_message, image_urls)
        return

    # Watched channel logic
    channel_id = str(message.channel.id)
    if await is_channel_watched(channel_id):
        user_message = message.content

        # Direct trigger => always
        if contains_direct_trigger(user_message):
            await queue_aggregated_response(message, user_message, image_urls)
            return

        # Soft/random => maybe
        has_soft = contains_soft_trigger(user_message)
        random_roll = random.random() < PERSONA.random_response_chance
        if has_soft or random_roll:
            if not local_should_consider_responding(message, user_message):
        return

            if await should_agent_respond(user_message):
                await queue_aggregated_response(message, user_message, image_urls)
                return
    await bot.process_commands(message)

@bot.command(name="ask")
async def cmd_ask(ctx, *, user_message: str):
    image_urls = extract_image_urls(ctx.message)
    await handle_agent_response(ctx.message, user_message, image_urls)

@bot.command(name="watch")
async def cmd_watch(ctx):
    if OWNER_ID and ctx.author.id != OWNER_ID:
        return
    channel_id = str(ctx.channel.id)
    WATCHED_CHANNEL_IDS.add(channel_id)
    async with db_conn() as conn:
        await conn.execute(
            "INSERT INTO watched_channels (channel_id) VALUES ($1) ON CONFLICT (channel_id) DO NOTHING",
            channel_id,
        )
    await ctx.reply(PERSONA.watch_on_message)

@bot.command(name="unwatch")
async def cmd_unwatch(ctx):
    if OWNER_ID and ctx.author.id != OWNER_ID:
        return
    channel_id = str(ctx.channel.id)
    WATCHED_CHANNEL_IDS.discard(channel_id)
    async with db_conn() as conn:
        await conn.execute("DELETE FROM watched_channels WHERE channel_id = $1", channel_id)
    await ctx.reply(PERSONA.watch_off_message)

@bot.command(name="state")
async def cmd_state(ctx):
    if OWNER_ID and ctx.author.id != OWNER_ID:
        return

    channel_id = str(ctx.channel.id)
    watched = await is_channel_watched(channel_id)

    msg = (
        f"STATE\n"
        f"- watched_channel: {watched}\n"
        f"- model_main: {MODEL_MAIN}\n"
        f"- model_side: {MODEL_SIDE}\n"
        f"- context_messages: {len(get_context_messages(channel_id, str(ctx.author.id)))}\n"
        f"- random_chance: {PERSONA.random_response_chance}\n"
        f"- images_enabled: {ENABLE_IMAGES}\n"
    )

    await ctx.reply(f"```{msg}```")

# ============================================================
# Run
# ============================================================

bot.run(DISCORD_TOKEN)
