# websocket_server.py

import asyncio
import websockets
import openai
import os
from dotenv import load_dotenv
from io import BytesIO
import json
import wave
import time

load_dotenv()
PORT = 8765
CHANNELS = 1
RATE = 16000
SAMPLE_WIDTH = 2

# Models - Use faster models for lower latency
TRANSCRIPTION_MODEL = "whisper-1"
CHAT_MODEL = "gpt-4-turbo"
TTS_MODEL = "tts-1"
TTS_RESPONSE_FORMAT = "pcm"

# --- NEW: VAD (Voice Activity Detection) Constants ---
# How long a pause to wait for before transcribing an interim phrase.
INTERIM_PAUSE_THRESHOLD = 0.8  # seconds
# How long a pause to wait for before considering the user's turn to be fully complete.
FINAL_PAUSE_THRESHOLD = 2.0  # seconds

client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print(f"WebSocket server starting on port {PORT}...")

class ConnectionPipeline:
    def __init__(self, websocket):
        self.websocket = websocket
        self.voice = "alloy"
        self.system_prompt = "You are a helpful assistant."
        self.conversation_history = []
        # --- NEW: State to manage the user's current turn ---
        self.current_user_turn = ""
        self.pipeline_tasks = []
        self.is_interrupted = asyncio.Event()

        # Queues for inter-task communication
        self.audio_in_queue = asyncio.Queue()
        self.transcript_queue = asyncio.Queue()
        self.llm_out_queue = asyncio.Queue()
        self.tts_out_queue = asyncio.Queue()

    def start(self, config):
        self.voice = config.get("voice", "alloy")
        self.system_prompt = config.get("system_prompt", "You are a helpful assistant.")
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]
        self.current_user_turn = "" # Reset on start
        
        greeting = config.get("greeting")
        if greeting:
            self.conversation_history.append({"role": "assistant", "content": greeting})
            asyncio.create_task(self.send_greeting(greeting))

        self.pipeline_tasks.append(asyncio.create_task(self.handle_client_input()))
        self.pipeline_tasks.append(asyncio.create_task(self.transcribe_audio())) # <-- This function is heavily modified
        self.pipeline_tasks.append(asyncio.create_task(self.generate_llm_response())) # <-- This function is slightly modified
        self.pipeline_tasks.append(asyncio.create_task(self.stream_tts_audio()))
        self.pipeline_tasks.append(asyncio.create_task(self.send_audio_to_client()))

    async def send_greeting(self, greeting_text):
        await self.websocket.send(json.dumps({"type": "text", "data": greeting_text}))
        try:
            response = await client.audio.speech.create(
                model=TTS_MODEL, voice=self.voice, input=greeting_text, response_format=TTS_RESPONSE_FORMAT
            )
            for chunk in response.iter_bytes(chunk_size=1024):
                await self.websocket.send(chunk)
        except Exception as e:
            print(f"Error sending greeting: {e}")

    async def handle_client_input(self):
        try:
            async for message in self.websocket:
                if isinstance(message, bytes):
                    await self.audio_in_queue.put(message)
                elif isinstance(message, str):
                    msg_data = json.loads(message)
                    if msg_data.get("type") == "interrupt":
                        print("Pipeline interrupted by client.")
                        self.is_interrupted.set()
                    elif msg_data.get("type") == "text_input":
                        user_text = msg_data.get("data")
                        print(f"User typed: {user_text}")
                        self.conversation_history.append({"role": "user", "content": user_text})
                        await self.transcript_queue.put(user_text)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.stop()

    # --- MODIFIED: transcribe_audio ---
    # This function now performs streaming transcription with interim and final pauses.
    async def transcribe_audio(self):
        audio_buffer = bytearray()
        last_speech_time = time.time()
        
        while True:
            try:
                # Consume audio from the queue with a short timeout
                chunk = await asyncio.wait_for(self.audio_in_queue.get(), timeout=0.1)
                audio_buffer.extend(chunk)
                last_speech_time = time.time()
            except asyncio.TimeoutError:
                now = time.time()
                is_speaking = len(audio_buffer) > RATE // 4 # Basic check if there's enough audio to be speech
                
                # Condition 1: Interim pause. User paused but might continue.
                if is_speaking and (now - last_speech_time > INTERIM_PAUSE_THRESHOLD):
                    print("Detected interim pause, transcribing phrase...")
                    await self.process_and_transcribe_chunk(audio_buffer)
                    audio_buffer.clear() # Clear buffer after processing

                # Condition 2: Final pause. User is likely done with their turn.
                elif not is_speaking and self.current_user_turn and (now - last_speech_time > FINAL_PAUSE_THRESHOLD):
                    print("Detected final pause, finalizing user's turn.")
                    # Add the complete turn to conversation history
                    self.conversation_history.append({"role": "user", "content": self.current_user_turn.strip()})
                    self.current_user_turn = "" # Reset for the next turn
                    audio_buffer.clear() # Ensure buffer is clear

                # Condition 3: Noise. Clear buffer if it's just small amounts of noise.
                elif not is_speaking and not self.current_user_turn and len(audio_buffer) > 0:
                    audio_buffer.clear()


    async def process_and_transcribe_chunk(self, audio_data):
        """Helper to transcribe a chunk of audio and feed it to the LLM."""
        if not audio_data: return
        
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS); wf.setsampwidth(SAMPLE_WIDTH); wf.setframerate(RATE)
            wf.writeframes(audio_data)
        
        try:
            transcript = await client.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL, 
                file=("audio.wav", wav_buffer.getvalue()), 
                language="en"
            )
            user_text = transcript.text.strip()
            if user_text:
                print(f"User said (interim): {user_text}")
                self.current_user_turn += " " + user_text
                await self.transcript_queue.put(user_text) # Send this phrase to the LLM
        except Exception as e:
            print(f"Transcription error: {e}")

    # --- MODIFIED: generate_llm_response ---
    # This function now uses a dynamically constructed message history for each LLM call.
    async def generate_llm_response(self):
        while True:
            transcript_phrase = await self.transcript_queue.get()
            self.is_interrupted.clear()
            
            print("Generating AI response...")
            full_response = ""
            
            # Construct a temporary history for the API call
            messages_for_api = self.conversation_history + [{"role": "user", "content": transcript_phrase}]

            try:
                response_stream = await client.chat.completions.create(
                    model=CHAT_MODEL, messages=messages_for_api, stream=True
                )
                async for chunk in response_stream:
                    if self.is_interrupted.is_set():
                        print("LLM generation cancelled.")
                        await response_stream.aclose()
                        break
                    
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_response += delta
                        await self.llm_out_queue.put(delta)
                
                if not self.is_interrupted.is_set() and full_response:
                    # Append full AI response to history only after it's complete
                    self.conversation_history.append({"role": "assistant", "content": full_response.strip()})
                await self.llm_out_queue.put(None) # Sentinel for end of response

            except Exception as e:
                print(f"LLM error: {e}")

    async def stream_tts_audio(self):
        """Takes text chunks from the LLM, generates audio, and puts it in the final queue."""
        while True:
            sentence_buffer = ""
            while True:
                text_chunk = await self.llm_out_queue.get()
                if self.is_interrupted.is_set():
                    print("TTS generation cancelled.")
                    while not self.llm_out_queue.empty(): self.llm_out_queue.get_nowait()
                    sentence_buffer = ""
                    break

                if text_chunk is None:
                    if sentence_buffer:
                        await self.generate_and_queue_audio(sentence_buffer)
                    await self.tts_out_queue.put(None)
                    sentence_buffer = ""
                    break
                
                sentence_buffer += text_chunk
                if any(p in sentence_buffer for p in ".?!"):
                    await self.generate_and_queue_audio(sentence_buffer)
                    sentence_buffer = ""

    async def generate_and_queue_audio(self, text):
        if not text.strip(): return
        
        print(f"Streaming speech for: '{text.strip()}'")
        await self.websocket.send(json.dumps({"type": "text", "data": text.strip()}))
        try:
            response = await client.audio.speech.create(
                model=TTS_MODEL, voice=self.voice, input=text, response_format=TTS_RESPONSE_FORMAT
            )
            for chunk in response.iter_bytes(chunk_size=1024):
                if self.is_interrupted.is_set():
                    print("Audio chunking interrupted.")
                    break
                await self.tts_out_queue.put(chunk)
        except Exception as e:
            print(f"TTS generation error: {e}")

    async def send_audio_to_client(self):
        while True:
            chunk = await self.tts_out_queue.get()
            if chunk is None: continue
            
            try:
                await self.websocket.send(chunk)
            except websockets.exceptions.ConnectionClosed:
                break

    def stop(self):
        print("Stopping connection pipeline...")
        for task in self.pipeline_tasks:
            task.cancel()

async def audio_handler(websocket, path=None):
    print("Client connected.")
    config_message = await websocket.recv()
    config = json.loads(config_message)

    pipeline = ConnectionPipeline(websocket)
    pipeline.start(config)

    try:
        await asyncio.gather(*pipeline.pipeline_tasks)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    finally:
        pipeline.stop()

async def main():
    async with websockets.serve(audio_handler, "localhost", PORT, max_size=None):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shut down.")