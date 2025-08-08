# audio.py

import asyncio
import websockets
import pyaudio
import threading
import queue
import time
import json

class RealtimeAudioClient:
    def __init__(self, text_queue):
        self.text_update_queue = text_queue
        self.is_running = threading.Event()
        self.is_recording = threading.Event()
        self.is_speaking = False
        self.connection_active = False
        self.main_thread = None
        self.speaker_thread = None
        self.pyaudio_instance = pyaudio.PyAudio()
        self.control_queue = None
        self.loop = None
        self.playback_queue = queue.Queue()
        self.speaker_stream = None
        self.URI = "ws://localhost:8765"
        self.CHUNK = 1024
        self.CHANNELS = 1
        self.RATE = 16000
        self.FORMAT = pyaudio.paInt16

    def start(self, config_message):
        if not self.connection_active:
            print("[AudioClient] Starting connection...")
            self.connection_active = True
            self.is_running.set()
            with self.playback_queue.mutex:
                self.playback_queue.queue.clear()
            self.main_thread = threading.Thread(target=self._run_connection_loop, args=(config_message,))
            self.main_thread.start()

    def stop(self):
        if self.connection_active:
            print("[AudioClient] Stopping connection...")
            # FIX: Immediately clear the playback queue to silence any lingering audio
            with self.playback_queue.mutex:
                self.playback_queue.queue.clear()
            
            self.is_running.clear()
            if self.main_thread and self.main_thread.is_alive():
                self.main_thread.join(timeout=2)
            if self.speaker_thread and self.speaker_thread.is_alive():
                self.speaker_thread.join(timeout=2)
            self.connection_active = False
            print("[AudioClient] Connection stopped.")

    def start_recording(self): self.is_recording.set()
    def stop_recording(self): self.is_recording.clear()

    def send_interrupt(self):
        if self.connection_active and self.loop and self.control_queue:
            with self.playback_queue.mutex:
                self.playback_queue.queue.clear()
            self.is_speaking = False
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.control_queue.put_nowait, {"type": "interrupt"})
    
    def send_text(self, text):
        if self.connection_active and self.loop and self.control_queue:
            message = {"type": "text_input", "data": text}
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.control_queue.put_nowait, message)

    def _run_connection_loop(self, config_message):
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.control_queue = asyncio.Queue()
            self.loop.run_until_complete(self._process_audio(config_message))
        except Exception as e:
            print(f"[AudioClient] Connection loop failed: {e}")
        finally:
            if self.loop.is_running(): self.loop.close()
            self.loop = None; self.control_queue = None

    async def _process_audio(self, config_message):
        try:
            async with websockets.connect(self.URI, max_size=None) as websocket:
                await websocket.send(json.dumps(config_message))
                self.speaker_thread = threading.Thread(target=self._speaker_thread, daemon=True)
                self.speaker_thread.start()
                await asyncio.gather(
                    self._control_sender(websocket),
                    self._audio_sender(websocket),
                    self._audio_receiver(websocket)
                )
        except Exception as e:
            print(f"[AudioClient] Processing failed: {e}")
        finally:
            self.connection_active = False

    async def _control_sender(self, websocket):
        while self.is_running.is_set():
            try:
                msg = await asyncio.wait_for(self.control_queue.get(), timeout=0.1)
                await websocket.send(json.dumps(msg))
            except asyncio.TimeoutError: continue
            except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed): break

    async def _audio_sender(self, websocket):
        mic_stream = self.pyaudio_instance.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
        while self.is_running.is_set():
            if self.is_recording.is_set():
                try:
                    data = mic_stream.read(self.CHUNK, exception_on_overflow=False)
                    if self.is_running.is_set(): await websocket.send(data)
                except websockets.exceptions.ConnectionClosed: break
            else:
                await asyncio.sleep(0.01)
        mic_stream.stop_stream(); mic_stream.close()

    async def _audio_receiver(self, websocket):
        while self.is_running.is_set():
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                if isinstance(message, str):
                    msg_data = json.loads(message)
                    if msg_data.get("type") == "text": self.text_update_queue.put(msg_data.get("data"))
                elif isinstance(message, bytes): self.playback_queue.put(message)
            except asyncio.TimeoutError: continue
            except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed): break

    def _speaker_thread(self):
        self.speaker_stream = self.pyaudio_instance.open(format=pyaudio.paInt16, channels=self.CHANNELS, rate=24000, output=True)
        while self.is_running.is_set():
            try:
                chunk = self.playback_queue.get(timeout=0.1)
                if self.speaker_stream and self.speaker_stream.is_active():
                    self.speaker_stream.write(chunk)
                if not self.is_speaking: self.is_speaking = True
            except queue.Empty:
                if self.is_speaking: self.is_speaking = False
            except OSError:
                break
        if self.speaker_stream and self.speaker_stream.is_active():
            self.speaker_stream.stop_stream(); self.speaker_stream.close()

    def shutdown(self):
        self.stop(); self.pyaudio_instance.terminate()