import time
import asyncio
import aiosqlite
from openai import OpenAI
import os
from datetime import datetime
import requests
import logging
import json
from settings import settings
import sqlite3

# Initialize the OpenAI client with Assistant v2
client = OpenAI(api_key=settings["openAIToken"])

async def init_db():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, settings["DB"])

    async with aiosqlite.connect(db_path) as conn:
        await conn.execute('''CREATE TABLE IF NOT EXISTS threads (
                                chat_id INTEGER,
                                thread_id TEXT,
                                user_id TEXT)''')
        await conn.execute('''CREATE TABLE IF NOT EXISTS conversations (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                chat_id INTEGER,
                                user_id TEXT,
                                thread_id TEXT,
                                assistant_id TEXT,
                                sender TEXT,
                                message TEXT,
                                message_id TEXT,
                                timestamp TEXT)''')
        await conn.commit()

async def blind_response(prompt):
    try:
        assistant_id = settings["assistant_id"]
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=[{"type": "text", "text": prompt}]
        )
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status in ["completed", "expired"]:
                break
            await asyncio.sleep(3)
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        return messages.data[0].content[0].text.value
    except Exception as e:
        logging.error(f"Error in blind_response: {str(e)}")
        return None

async def whisper_transcribe(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="en"
            )
        return transcript
    except Exception as e:
        logging.error(f"Error in whisper_transcribe: {e}")
        return ""

async def get_thread_id_and_user_id(chat_id, db_connection):
    async with db_connection.execute(
        'SELECT thread_id, user_id FROM threads WHERE chat_id = ? ORDER BY ROWID DESC LIMIT 1',
        (chat_id,)
    ) as cursor:
        result = await cursor.fetchone()
    return result if result else (None, None)

async def save_user_and_thread_id(chat_id, user_id, thread_id, db_connection):
    await db_connection.execute(
        'INSERT OR REPLACE INTO threads (chat_id, user_id, thread_id) VALUES (?, ?, ?)', 
        (chat_id, user_id, thread_id)
    )
    await db_connection.commit()

async def create_new_thread(chat_id, db_connection):
    try:
        thread = client.beta.threads.create()
        new_thread_id = thread.id
    except Exception as e:
        logging.error(f"Error creating new thread: {e}")
        return None, None
    
    user_id = f"User{chat_id}"
    await save_user_and_thread_id(chat_id, user_id, new_thread_id, db_connection)
    return new_thread_id, user_id

async def GPT_response(prompt, chat_id, db_connection, assistant_id=None, thread_id=None):
    if thread_id is not None:
        # Use the provided thread_id directly
        user_id = f"User{chat_id}"
    else:
        # Fallback: look up (legacy behavior)
        thread_id, user_id = await get_thread_id_and_user_id(chat_id, db_connection)
        if not thread_id:
            thread_id, user_id = await create_new_thread(chat_id, db_connection)

    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=[{"type": "text", "text": prompt}]
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )
    
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if run_status.status in ["completed", "expired"]:
            break
        await asyncio.sleep(3)
    
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    ai_response = messages.data[0].content[0].text.value
    return ai_response

    
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    # Ensure response is properly parsed
    ai_response = messages.data[0].content[0].text.value
    return ai_response
    #try:
    #    return json.loads(ai_response)  # Convert response from string to dict
    #except json.JSONDecodeError:
    #    logging.error(f"Error parsing AI response: {ai_response}")
    #    return {"error": "Invalid AI response format"}

