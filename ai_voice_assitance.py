import os
import json
import base64
import asyncio
import websockets
import threading
import uuid
import queue
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from azure.cognitiveservices.speech import AutoDetectSourceLanguageConfig
from azure.cognitiveservices.speech import AudioStreamWaveFormat
# Process with LLM
from chat_helper import generate_stream
import torch
import numpy as np
from silero_vad import load_silero_vad

load_dotenv()

# Configuration
account_sid = os.getenv("TWILIO_ACCOUNT_SID_CALL")
auth_token = os.getenv("TWILIO_AUTH_TOKEN_CALL")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SMS_KEY = os.getenv('SMS_KEY')
PORT = int(os.getenv('PORT', 8010))

# ElevenLabs configuration
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL")

# Current date/time
current_datetime = datetime.now()
current_date = current_datetime.strftime("%d-%m-%Y")
current_time = current_datetime.strftime("%I:%M %p")

app = FastAPI()

vad_model = None
# Validate environment variables
if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
    raise ValueError('Missing the Azure Speech keys. Please set them in the .env file.')

if not OPENAI_API_KEY:
    raise ValueError('Missing OpenAI API key. Please set it in the .env file.')

if not ELEVENLABS_API_KEY:
    raise ValueError('Missing ElevenLabs API key. Please set it in the .env file.')

print(f"Using Azure Speech Region: {AZURE_SPEECH_REGION}")
print(f"Speech Key is present: {'Yes' if AZURE_SPEECH_KEY else 'No'}")
print(f"OpenAI API Key is present: {'Yes' if OPENAI_API_KEY else 'No'}")
print(f"ElevenLabs API Key is present: {'Yes' if ELEVENLABS_API_KEY else 'No'}")

# Initialize the LLM
llm = ChatOpenAI(api_key=OPENAI_API_KEY,streaming=True,temperature=0.3,model="gpt-4o-mini")

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Hindi Voice Assistant Server is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    print("Received incoming call request")
    body = await request.body()
    print("Headers:", request.headers)
    print("Body:", body.decode())
    
    response = VoiceResponse()
    host = request.url.hostname
    print(f"Using hostname for stream: {host}")
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

# Thread-safe shared state
class SharedState:
    def __init__(self):
        # Queues for data passing between threads
        self.input_audio_queue = queue.Queue()  # Raw audio → STT
        self.input_vad_queue = queue.Queue()    # Raw audio → VAD (new dedicated queue)
        self.recognized_text_queue = queue.Queue()  # STT → LLM
        self.response_text_queue = queue.Queue()  # LLM → TTS
        self.audio_chunk_queue = queue.Queue()  # TTS → Caller
        
        # Events for thread coordination
        self.interrupt_event = threading.Event()  # Signal when caller speaks
        self.shutdown_event = threading.Event()  # Signal for graceful shutdown
        self.interruption_handled = False  # Track if we've already handled interruption for current speech
        
        # Status tracking
        self.processing_status = {
            "stt": "idle",
            "llm": "idle",
            "tts": "idle",
            "caller": "idle",
            "vad": "idle"
        }
        self.error_queue = queue.Queue()
        
        # Connection data
        self.stream_sid = None
        self.call_sid = None
        
        # Thread-safe access
        self._lock = threading.RLock()
    
    def update_status(self, component, status):
        with self._lock:
            self.processing_status[component] = status
            print(f"STATUS UPDATE: {component} → {status}")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections using four-thread parallel architecture"""
    print("Client connected to WebSocket")
    await websocket.accept()
    
    # Initialize shared state
    shared_state = SharedState()
    
    # Initialize the event loop for the main thread
    loop = asyncio.get_running_loop()
    
    # Create an event to coordinate thread shutdown from async context
    async_shutdown_trigger = asyncio.Event()
    
    # Thread 1: Caller Interface Thread
    def caller_interface_thread():
        """Handles bidirectional audio with the caller"""
        shared_state.update_status("caller", "active")
        
        try:
            # This function will be called by the main thread to send audio to the caller
            async def send_audio_to_caller():
                last_chunk_time = None
                
                while not shared_state.shutdown_event.is_set():
                    try:
                        # Check for interrupt BEFORE getting audio from the queue
                        if shared_state.interrupt_event.is_set():
                            # IMPORTANT: Don't just continue, CLEAR the queue!
                            while not shared_state.audio_chunk_queue.empty():
                                try:
                                    shared_state.audio_chunk_queue.get_nowait()
                                    shared_state.audio_chunk_queue.task_done()
                                except queue.Empty:
                                    break
                            await asyncio.sleep(0.001)
                            continue
                        
                        # Non-blocking check for new audio chunks
                        try:
                            chunk_with_metadata = shared_state.audio_chunk_queue.get_nowait()
                        except queue.Empty:
                            await asyncio.sleep(0.01)
                            continue
                        
                        # Check AGAIN for interrupt before sending
                        if shared_state.interrupt_event.is_set():
                            # Just discard this chunk without sending it
                            shared_state.audio_chunk_queue.task_done()
                            continue
                        
                        # Extract actual audio data and metadata
                        if isinstance(chunk_with_metadata, dict) and "audio_data" in chunk_with_metadata:
                            # New format with metadata
                            chunk = chunk_with_metadata["audio_data"]
                            chunk_duration_ms = chunk_with_metadata.get("duration_ms", 200)
                            is_final = chunk_with_metadata.get("is_final", False)
                        else:
                            # Old format (raw bytes)
                            chunk = chunk_with_metadata
                            chunk_duration_ms = 200  # default assumption
                            is_final = False
                        
                        # Send the audio chunk to the client
                        payload = base64.b64encode(chunk).decode('utf-8')
                        audio_delta = {
                            "event": "media",
                            "streamSid": shared_state.stream_sid,
                            "media": {
                                "payload": payload
                            }
                        }
                        await websocket.send_json(audio_delta)
                        
                        # Generate a unique ID for this chunk
                        chunk_id = str(uuid.uuid4())[:8]
                        await send_mark_message(websocket, shared_state.stream_sid, chunk_id)
                        
                        # Log the sent chunk
                        print(f"Sent audio chunk to caller - length: {len(chunk)} bytes (~{chunk_duration_ms}ms), mark_id: chunk_{chunk_id}")
                        
                        shared_state.audio_chunk_queue.task_done()
                        
                        # Natural pacing - sleep approximately the time it would take to play the audio
                        # This makes interruptions feel more natural (but slightly reduced to stay responsive)
                        sleep_time = chunk_duration_ms / 1000 * 0.9  # 90% of actual duration to stay responsive
                        await asyncio.sleep(sleep_time)
                        
                        # Record this chunk's time for calculating pauses
                        last_chunk_time = time.time()
                        
                    except asyncio.CancelledError:
                        print("Audio sender task cancelled")
                        break
                    except Exception as e:
                        print(f"Error in audio sender: {str(e)}")
                        shared_state.error_queue.put(("caller", str(e)))
                            
            # Create and start the audio sender task on the main event loop
            sender_task = asyncio.run_coroutine_threadsafe(send_audio_to_caller(), loop)
            
            # Wait until we need to shut down
            while not shared_state.shutdown_event.is_set():
                time.sleep(0.1)
                    
        except Exception as e:
            print(f"Error in caller interface thread: {str(e)}")
            shared_state.error_queue.put(("caller", str(e)))
        finally:
            shared_state.update_status("caller", "terminated")
    
    # Thread 2: Speech to Text Thread
    def speech_to_text_thread():
        """Converts audio to Hindi text with English word support"""
        shared_state.update_status("stt", "initializing")
        
        try:
            # Configure speech recognition with bilingual settings
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
            
            # speech_config.speech_recognition_language = "hi-IN"
            from azure.cognitiveservices.speech import AutoDetectSourceLanguageConfig
            auto_detect_source_language_config = AutoDetectSourceLanguageConfig(
                languages=["hi-IN", "en-In"]
            )
            
            # Audio format for Twilio's audio stream (mulaw 8kHz)
            audio_format = speechsdk.audio.AudioStreamFormat(samples_per_second=8000, 
                                                        bits_per_sample=8,
                                                        channels=1,wave_stream_format=AudioStreamWaveFormat.MULAW)
            
            push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
            audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
            
            # More aggressive settings for faster results
            # speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "500")
            speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "300") 
            speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "200")
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, 
                                                            auto_detect_source_language_config=auto_detect_source_language_config,
                                                        audio_config=audio_config)
            shared_state.update_status("stt", "active")
            print("Speech recognition initialized with Hindi-English bilingual settings")
            
            # Speech recognition callbacks
            def recognized_cb(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    text = evt.result.text.strip()
                    
                    # Skip empty results
                    if not text:
                        return
                    
                    print(f"RECOGNIZED TEXT: {text}")
                    
                    # Queue the recognized text for LLM processing without interrupt handling
                    shared_state.recognized_text_queue.put(text)
                    print(f"Queued text for LLM processing: {text}")
                    
                elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                    print("SPEECH NOT RECOGNIZED: No match found")

            def recognizing_cb(evt):
                """Callback for recognizing events (interim results)"""
                interim_text = evt.result.text
                if interim_text:
                    print(f"RECOGNIZING: {interim_text}")
                    
                    

            def canceled_cb(evt):
                """Callback for canceled speech recognition"""
                print(f"CANCELED: Reason={evt.reason}")
                if evt.reason == speechsdk.CancellationReason.Error:
                    error_details = evt.error_details
                    print(f"ERROR: {error_details}")
                    shared_state.error_queue.put(("stt", error_details))
                    
                    # Attempt to restart recognition after error
                    try:
                        speech_recognizer.stop_continuous_recognition_async()
                        time.sleep(0.5)
                        speech_recognizer.start_continuous_recognition_async()
                        print("Restarted speech recognition after error")
                    except Exception as e:
                        print(f"Failed to restart recognition: {str(e)}")

            # Connect callbacks to the recognizer
            speech_recognizer.recognized.connect(recognized_cb)
            speech_recognizer.recognizing.connect(recognizing_cb)
            speech_recognizer.canceled.connect(canceled_cb)
            
            # Start recognition
            speech_recognizer.start_continuous_recognition_async()
            print("Continuous speech recognition started")
            
            # Process audio from the queue
            while not shared_state.shutdown_event.is_set():
                try:
                    # Get audio data from the queue with timeout
                    audio_bytes = shared_state.input_audio_queue.get(timeout=0.001)
                    
                    # Send directly to Azure without buffering
                    push_stream.write(bytes(audio_bytes))
                    
                    shared_state.input_audio_queue.task_done()
                    
                    # IMPORTANT: Only clear the interrupt flag AFTER successfully 
                    # processing the audio - this was in the wrong place before
                    if shared_state.interrupt_event.is_set():
                        print("New speech detected - maintaining interrupt state")
                    
                except queue.Empty:
                    # Queue is empty, just continue
                    continue
                except Exception as e:
                    print(f"Error in STT processing: {str(e)}")
                    shared_state.error_queue.put(("stt", str(e)))
            
            speech_recognizer.stop_continuous_recognition_async()
            print("Speech recognition stopped")
            
        except Exception as e:
            print(f"Error in speech-to-text thread: {str(e)}")
            shared_state.error_queue.put(("stt", str(e)))
        finally:
            shared_state.update_status("stt", "terminated")
    
    # Thread 3: LLM Processing Thread
    def llm_processing_thread():
        """Process recognized text and generate responses with streaming"""
        shared_state.update_status("llm", "active")
        
        try:
            while not shared_state.shutdown_event.is_set():
                # Check for interrupt before processing
                if shared_state.interrupt_event.is_set():
                    # Clear any previous responses since we're being interrupted
                    while not shared_state.response_text_queue.empty():
                        try:
                            shared_state.response_text_queue.get_nowait()
                            shared_state.response_text_queue.task_done()
                        except queue.Empty:
                            break
                
                try:
                    # Get text from the queue with timeout
                    text = shared_state.recognized_text_queue.get()
                    shared_state.update_status("llm", "processing")
                    
                    print(f"LLM processing: {text}")
                    
                    # Reset interrupt flag before processing
                    shared_state.interrupt_event.clear()
                    
                    try:
                        
                        # Create a coroutine for streaming LLM responses
                        async def process_stream():
                            # Get streaming response from the LLM using stream_sid as session ID
                            stream = await generate_stream(text, session_id=shared_state.stream_sid, history_size=10)
                            
                            # Process streaming chunks
                            async for chunk in stream:
                                # Check if interrupted
                                if shared_state.interrupt_event.is_set():
                                    print("LLM streaming interrupted - stopping generation")
                                    return
                                
                                if chunk.content:
                                    # Check for interruption before queuing
                                    if shared_state.interrupt_event.is_set():
                                        print("LLM interrupted before queuing chunk")
                                        return
                                    
                                    # Send chunk directly to TTS without buffering
                                    if chunk.content.strip():  # Skip empty chunks
                                        print(f"Sending chunk directly to TTS: {chunk.content}")
                                        shared_state.response_text_queue.put(chunk.content)
                        
                        # Run the streaming process in the event loop
                        stream_future = asyncio.run_coroutine_threadsafe(process_stream(), loop)
                        
                        # Wait for stream completion
                        stream_future.result()  # This blocks until stream is done
                        
                        print("Stream completed, sending Flush to TTS")
                        shared_state.response_text_queue.put("__finish__")  # Send finish marker only after completion
                        # Wait for completion or interruption
                        while not stream_future.done() and not shared_state.shutdown_event.is_set():
                            if shared_state.interrupt_event.is_set():
                                print("LLM waiting interrupted - cancelling future")
                                stream_future.cancel()
                                break
                            time.sleep(0.001)  # Small sleep to prevent CPU spinning
                            
                    except Exception as e:
                        print(f"Error in LLM processing: {str(e)}")
                        shared_state.error_queue.put(("llm", str(e)))
                    
                    shared_state.recognized_text_queue.task_done()
                    shared_state.update_status("llm", "active")
                    
                except queue.Empty:
                    # Queue is empty, just continue
                    continue
        
        except Exception as e:
            print(f"Error in LLM thread: {str(e)}")
            shared_state.error_queue.put(("llm", str(e)))
        finally:
            shared_state.update_status("llm", "terminated")
    
    # Thread 4: TTS Generation Thread
    def tts_generation_thread():
        """Convert text to speech using ElevenLabs with streaming support and buffering"""
        shared_state.update_status("tts", "active")
        
        try:
            # Add buffer metadata to track timing
            buffer_metadata = {
                "chunk_duration_ms": 200,  # Target duration for each chunk
                "samples_per_second": 8000,  # 8kHz audio
                "bytes_per_sample": 1      # 8-bit ulaw
            }
            
            # Calculate bytes per 200ms chunk
            bytes_per_chunk = int((buffer_metadata["chunk_duration_ms"] / 1000) * 
                                   buffer_metadata["samples_per_second"] * 
                                   buffer_metadata["bytes_per_sample"])
            
            print(f"TTS buffering: targeting ~{buffer_metadata['chunk_duration_ms']}ms chunks " 
                  f"({bytes_per_chunk} bytes per chunk)")
            
            # Function to process TTS in a single WebSocket connection
            async def process_tts_stream():
                # ElevenLabs WebSocket URI
                uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream-input?model_id={ELEVENLABS_MODEL}&output_format=ulaw_8000&inactivity_timeout=180"
                
                try:
                    async with websockets.connect(uri, extra_headers={"xi-api-key": ELEVENLABS_API_KEY}) as ws:
                        # Initialize with voice settings - optimize for low latency
                        await ws.send(json.dumps({
                            "text": " ",
                            "voice_settings": {
                                "stability": 0.4,
                                "similarity_boost": 0.7,
                                "style": 0.0,
                                "use_speaker_boost": True,
                                "speed": 1.0
                            }
                        }))
                        
                        # Create task to listen for audio responses with buffering
                        audio_receiver_task = asyncio.create_task(receive_audio_chunks(ws, shared_state, buffer_metadata))
                        
                        # Keep WebSocket open and process chunks as they arrive
                        while not shared_state.shutdown_event.is_set():
                            try:
                                # Non-blocking check for interruption
                                if shared_state.interrupt_event.is_set():
                                    print("TTS interrupted - stopping stream")
                                    break
                                    
                                # Get text from queue with short timeout
                                try:
                                    text = shared_state.response_text_queue.get(timeout=0.001)
                                    
                                    # Handle the special finish marker
                                    if text == "__finish__":
                                        print("Received finish marker, flushing audio stream")
                                        # Send empty text with flush parameter to ensure final audio is generated
                                        await ws.send(json.dumps({
                                            "text": "",
                                            "flush": True
                                        }))
                                        shared_state.response_text_queue.task_done()
                                        continue
                                    
                                    # Skip empty text
                                    if not text:
                                        shared_state.response_text_queue.task_done()
                                        continue
                                        
                                    print(f"Sending text chunk to ElevenLabs: {text}")
                                    
                                    # Send the text chunk directly to the open WebSocket with try_trigger_generation
                                    await ws.send(json.dumps({
                                        "text": text,
                                        "try_trigger_generation": True  # Force immediate processing
                                    }))
                                    
                                    shared_state.response_text_queue.task_done()
                                    
                                except queue.Empty:
                                    # No text available, just continue the loop
                                    await asyncio.sleep(0.001)
                                    continue
                                    
                            except asyncio.CancelledError:
                                print("TTS processing cancelled")
                                break
                        
                        # Cancel the audio receiver task
                        if audio_receiver_task and not audio_receiver_task.done():
                            audio_receiver_task.cancel()
                        
                except Exception as e:
                    print(f"Error in TTS WebSocket: {str(e)}")
                    shared_state.error_queue.put(("tts", str(e)))
    
        except Exception as e:
            print(f"Error in TTS thread: {str(e)}")
            shared_state.error_queue.put(("tts", str(e)))
        finally:
            shared_state.update_status("tts", "terminated")
        
        # Helper function to receive and buffer audio chunks from the WebSocket
        async def receive_audio_chunks(ws, shared_state, buffer_metadata):
            try:
                audio_buffer = bytearray()
                bytes_per_chunk = int((buffer_metadata["chunk_duration_ms"] / 1000) * 
                                   buffer_metadata["samples_per_second"] * 
                                   buffer_metadata["bytes_per_sample"])
                
                async for message in ws:
                    # Check for interruption
                    if shared_state.interrupt_event.is_set():
                        return
                    
                    data = json.loads(message)
                    if "audio" in data and data["audio"] is not None:
                        # Decode audio data
                        audio_data = base64.b64decode(data["audio"])
                        
                        # Add to buffer
                        audio_buffer.extend(audio_data)
                        
                        # Process complete chunks (200ms each)
                        while len(audio_buffer) >= bytes_per_chunk:
                            # Extract a chunk
                            chunk = bytes(audio_buffer[:bytes_per_chunk])
                            audio_buffer = audio_buffer[bytes_per_chunk:]
                            
                            # Add chunk metadata
                            chunk_with_metadata = {
                                "audio_data": chunk,
                                "duration_ms": buffer_metadata["chunk_duration_ms"],
                                "timestamp": time.time()
                            }
                            
                            # Queue the chunk with metadata
                            shared_state.audio_chunk_queue.put(chunk_with_metadata)
                            print(f"Buffered audio chunk: {len(chunk)} bytes (~{buffer_metadata['chunk_duration_ms']}ms)")
                        
                    elif "isFinal" in data and data["isFinal"]:
                        # This indicates the end of a response
                        # Send any remaining buffer data
                        if len(audio_buffer) > 0:
                            # Calculate approximate duration for remaining data
                            remaining_duration = (len(audio_buffer) / bytes_per_chunk) * buffer_metadata["chunk_duration_ms"]
                            
                            # Queue the final chunk with metadata
                            chunk_with_metadata = {
                                "audio_data": bytes(audio_buffer),
                                "duration_ms": remaining_duration,
                                "timestamp": time.time(),
                                "is_final": True
                            }
                            
                            shared_state.audio_chunk_queue.put(chunk_with_metadata)
                            print(f"Buffered final audio chunk: {len(audio_buffer)} bytes (~{remaining_duration:.1f}ms)")
                            audio_buffer.clear()
                        
            except asyncio.CancelledError:
                # Normal cancellation
                pass
            except Exception as e:
                print(f"Error receiving audio chunks: {str(e)}")
        
        # Main loop - restart WebSocket connection if needed
        while not shared_state.shutdown_event.is_set():
            # Run the TTS processing with a single WebSocket connection
            tts_future = asyncio.run_coroutine_threadsafe(process_tts_stream(), loop)
            
            # Wait for completion or interruption
            while not tts_future.done() and not shared_state.shutdown_event.is_set():
                time.sleep(0.05)
            
            # Brief pause before reconnecting if needed
            if not shared_state.shutdown_event.is_set():
                time.sleep(0.2)
    
    # Remove the async function and replace with this thread function
    def vad_processing_thread():
        """Process audio with Voice Activity Detection in a dedicated thread"""
        shared_state.update_status("vad", "active")
        print("Starting Silero VAD processing thread")
        
        try:
            # Configuration
            window_size_samples = 256
            sample_rate = 8000
            speech_threshold = 0.75
            silence_threshold = 0.3
            speech_detected = False
            audio_buffer = np.array([], dtype=np.float32)
            
            # Get the VAD model
            model = get_vad_model()
            
            while not shared_state.shutdown_event.is_set():
                try:
                    # Get audio data from the dedicated VAD queue with timeout
                    try:
                        audio_bytes = shared_state.input_vad_queue.get(timeout=0.001)
                        
                        # Convert μ-law bytes to float array for VAD processing
                        audio_array = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32) / 255.0
                        
                        # Add new audio data to buffer
                        audio_buffer = np.append(audio_buffer, audio_array)
                        
                        # Process complete chunks
                        while len(audio_buffer) >= window_size_samples:
                            # Extract a chunk of exactly window_size_samples
                            chunk = audio_buffer[:window_size_samples]
                            # Remove the processed chunk from buffer
                            audio_buffer = audio_buffer[window_size_samples:]
                            
                            # Convert to PyTorch tensor
                            chunk_tensor = torch.tensor(chunk)
                            
                            # Process with VAD model
                            speech_prob = model(chunk_tensor, sample_rate).item()
                            
                            # Detect transitions
                            if not speech_detected and speech_prob > speech_threshold:
                                # Transition from silence to speech
                                speech_detected = True
                                print(f"VAD: Speech started (prob: {speech_prob:.2f})")
                                
                                # Signal interruption
                                if not shared_state.interruption_handled:
                                    shared_state.interrupt_event.clear()
                                    shared_state.interrupt_event.set()
                                    shared_state.interruption_handled = True
                                    
                                    # Queue the clear operation to run on the main thread
                                    asyncio.run_coroutine_threadsafe(
                                        handle_interruption(shared_state, websocket), 
                                        loop
                                    )
                            
                            elif speech_detected and speech_prob < silence_threshold:
                                # Transition from speech to silence
                                speech_detected = False
                                print(f"VAD: Speech ended (prob: {speech_prob:.2f})")
                                
                                # Reset interrupt flags
                                shared_state.interrupt_event.clear()
                                shared_state.interruption_handled = False
                        
                        shared_state.input_vad_queue.task_done()
                        
                    except queue.Empty:
                        # No audio data available, sleep briefly
                        time.sleep(0.001)
                        
                except Exception as e:
                    print(f"Error in VAD processing: {str(e)}")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Error in VAD thread: {str(e)}")
            shared_state.error_queue.put(("vad", str(e)))
        finally:
            shared_state.update_status("vad", "terminated")
    # Start all threads
    threads = []
    
    try:
        # Import the time module which is needed
        import time
        
        # Create and start all worker threads
        caller_thread = threading.Thread(target=caller_interface_thread, name="CallerThread")
        vad_thread = threading.Thread(target=vad_processing_thread, name="VADThread")
        stt_thread = threading.Thread(target=speech_to_text_thread, name="STTThread")
        llm_thread = threading.Thread(target=llm_processing_thread, name="LLMThread")
        tts_thread = threading.Thread(target=tts_generation_thread, name="TTSThread")

        threads = [caller_thread,vad_thread, stt_thread, llm_thread, tts_thread]
        
        for thread in threads:
            thread.daemon = True
            thread.start()
            print(f"Started {thread.name}")
        # Main websocket processing loop
        try:
            async for message in websocket.iter_text():
                data = json.loads(message)
                
                if data['event'] == 'media':
                    # Queue the audio data for both STT and VAD processing
                    audio_bytes = base64.b64decode(data['media']['payload'])
                    
                    shared_state.input_audio_queue.put(audio_bytes)
                    shared_state.input_vad_queue.put(audio_bytes)  # Add to VAD queue too
                    
                elif data['event'] == 'start':
                    # Store stream and call SIDs
                    shared_state.call_sid = data['start'].get('callSid', 'unknown')
                    shared_state.stream_sid = data['start']['streamSid']
                    print(f"Media stream started: CALL SID: {shared_state.call_sid}, STREAM SID: {shared_state.stream_sid}")
                    shared_state.interrupt_event.clear()
                    shared_state.interruption_handled = True  # Temporarily prevent interruption
                    # Add a small delay to ensure system is ready
                    await asyncio.sleep(1)
                    shared_state.response_text_queue.put("hello i am hospital ai assistant how can i assist you today ?")# Initial greeting
                    shared_state.response_text_queue.put("__finish__")
                    # After a short delay, allow interruptions again
                    await asyncio.sleep(2.0)  # Give enough time for greeting to be processed
                    shared_state.interruption_handled = False

                elif data['event'] == 'stop':
                    print("Media stream ended")
                    shared_state.shutdown_event.set()
                    break
                elif data['event'] == 'mark':
                    mark_name = data.get('mark', {}).get('name','')
                    print(f"Received mark event: {mark_name} - Audio chunk playback completed")
                # Check for errors
                while not shared_state.error_queue.empty():
                    try:
                        component, error = shared_state.error_queue.get_nowait()
                        print(f"ERROR in {component}: {error}")
                        shared_state.error_queue.task_done()
                    except queue.Empty:
                        break
                
        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            print(f"Error in main WebSocket loop: {str(e)}")
        
    except Exception as e:
        print(f"Error setting up threads: {str(e)}")
    finally:
        # Signal all threads to shut down
        print("Shutting down threads...")
        shared_state.shutdown_event.set()
        
        # Wait for threads to complete (with timeout)
        for thread in threads:
            thread.join(timeout=2.0)
            
        # Report thread status
        for thread in threads:
            if thread.is_alive():
                print(f"{thread.name} is still running")
            else:
                print(f"{thread.name} has terminated")

# Add this helper function outside the thread functions
async def handle_interruption(shared_state, websocket):
    """Handle interruption operations on the main thread with buffer awareness"""
    try:
        # Send clear message to Twilio - this stops audio playback on caller side
        await websocket.send_json({
            "event": "clear",
            "streamSid": shared_state.stream_sid
        })
        print(f"Sent CLEAR message from interruption handler")
        
        # More aggressive queue clearing with new queue format
        new_audio_queue = queue.Queue()
        new_response_queue = queue.Queue()
        
        # Safely replace queues with empty ones
        shared_state.audio_chunk_queue = new_audio_queue
        shared_state.response_text_queue = new_response_queue
        
        # Clear LLM queue to stop processing previous input
        while not shared_state.recognized_text_queue.empty():
            try:
                shared_state.recognized_text_queue.get_nowait()
                shared_state.recognized_text_queue.task_done()
            except queue.Empty:
                break
                
        print("Reset all queues to handle interruption")
    except Exception as e:
        print(f"Error in interruption handler: {str(e)}")

async def clear_queues(shared_state):
    """Clear queues asynchronously"""
    # Clear audio queue
    queue_items_cleared = 0
    while not shared_state.audio_chunk_queue.empty():
        try:
            shared_state.audio_chunk_queue.get_nowait()
            shared_state.audio_chunk_queue.task_done()
            queue_items_cleared += 1
        except queue.Empty:
            break
    print(f"Cleared {queue_items_cleared} items from audio queue")
    
    # Clear response text queue
    tts_items_cleared = 0
    while not shared_state.response_text_queue.empty():
        try:
            shared_state.response_text_queue.get_nowait()
            shared_state.response_text_queue.task_done()
            tts_items_cleared += 1
        except queue.Empty:
            break
    print(f"Cleared {tts_items_cleared} items from response queue")

async def send_mark_message(websocket, stream_sid, mark_id):
    """Send a mark message to track audio playback"""
    await websocket.send_json({
        "event": "mark",
        "streamSid": stream_sid,
        "mark": {
            "name": f"chunk_{mark_id}"
        }
    })
    print(f"Sent MARK message with ID: chunk_{mark_id}")

def get_vad_model():
    global vad_model
    if vad_model is None:
        print("Loading Silero VAD model...")
        vad_model = load_silero_vad()
        print("Silero VAD model loaded")
    return vad_model

if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")