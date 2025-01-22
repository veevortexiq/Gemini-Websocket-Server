## pip install --upgrade google-genai==0.3.0##
import asyncio
import json
import os
import websockets
from google import genai
import base64
from dotenv import load_dotenv
from google.cloud import speech
import queue
import threading
from google.cloud import speech_v1

load_dotenv()
# Load API key from environment
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
MODEL = "gemini-2.0-flash-exp"  # use your model ID

client = genai.Client(
  http_options={
    'api_version': 'v1alpha',
  }
)

class SpeechProcessor:
    def __init__(self):
        self.client = speech.SpeechClient()
        self.audio_queue = queue.Queue()
        self.is_running = True
        
        # Start processing thread
        self.thread = threading.Thread(target=self._process_audio, daemon=True)
        self.thread.start()
    
    def _process_audio(self):
        """Process audio in a separate thread"""
        while self.is_running:
            try:
                # Configure streaming recognition
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=24000,  # Matches Gemini's audio rate
                    language_code="en-US"
                )
                streaming_config = speech.StreamingRecognitionConfig(
                    config=config,
                    interim_results=True
                )

                # Generator for audio chunks
                def audio_generator():
                    yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
                    while self.is_running:
                        try:
                            chunk = self.audio_queue.get(timeout=1)
                            yield speech.StreamingRecognizeRequest(audio_content=chunk)
                        except queue.Empty:
                            continue

                # Process audio stream
                responses = self.client.streaming_recognize(requests=audio_generator())
                for response in responses:
                    for result in response.results:
                        transcript = result.alternatives[0].transcript
                        is_final = result.is_final
                        print(f"{'Final' if is_final else 'Interim'}: {transcript}")

            except Exception as e:
                print(f"Speech recognition error: {e}")
                continue

    def add_audio(self, audio_data):
        """Add audio chunk to processing queue"""
        self.audio_queue.put(audio_data)

    def close(self):
        """Clean up resources"""
        self.is_running = False
        self.thread.join(timeout=2)

async def gemini_session_handler(client_websocket: websockets.WebSocketServerProtocol):
    try:
        config_message = await client_websocket.recv()
        config_data = json.loads(config_message)
        config = config_data.get("setup", {})
        config["system_instruction"] = """You are a screen sharing AI assistant. Your core functions:
        Analyze Screen Content
        Describe what's visible
        Identify key elements
        Track important changes
        Provide Support
        Answer questions about visible content
        Explain unclear elements
        Guide users through processes
        Help with technical issues
        Communication Rules
        Be clear and concise
        Use professional tone
        Focus on relevant details
        Respond promptly
        Technical Monitoring
        Alert users to quality issues
        Suggest quick fixes
        Guide screen sharing setup
        Always prioritize clarity and efficiency in your responses.
        Maximum Size of your response 600 characters"""

        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print("Connected to Gemini API")

            async def send_to_gemini():
                try:
                    async for message in client_websocket:
                        try:
                            data = json.loads(message)
                            if "realtime_input" in data:
                                for chunk in data["realtime_input"]["media_chunks"]:
                                    if chunk["mime_type"] == "audio/pcm":
                                        await session.send({"mime_type": "audio/pcm", "data": chunk["data"]})
                                    elif chunk["mime_type"] == "image/jpeg":
                                        await session.send({"mime_type": "image/jpeg", "data": chunk["data"]})
                        except Exception as e:
                            reset_connections();
                            await client_websocket.send(json.dumps({
                                "error": str(e),
                                "status": "error"
                            }))
                            break
                finally:
                    print("send_to_gemini closed")

            async def receive_from_gemini():
                accumulated_responses = []
                accumulated_audio = bytearray()
                try:
                    while True:
                        print("receiving from gemini")
                        async for response in session.receive():
                            if response.server_content is None:
                                continue
                            
                            model_turn = response.server_content.model_turn
                            if model_turn:
                                for part in model_turn.parts:
                                    if hasattr(part, 'text') and part.text is not None:
                                        accumulated_responses.append({"text": part.text})
                                    elif hasattr(part, 'inline_data') and part.inline_data is not None:
                                        accumulated_audio.extend(part.inline_data.data)
                                        base64_audio = base64.b64encode(part.inline_data.data).decode('utf-8')
                                        await client_websocket.send(json.dumps({
                                            "audio": base64_audio,
                                        }))

                            if response.server_content.turn_complete:
                                if accumulated_audio:
                                    try:
                                        client = speech_v1.SpeechClient()
                                        audio = speech_v1.RecognitionAudio(content=bytes(accumulated_audio))
                                        config = speech_v1.RecognitionConfig(
                                            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                                            sample_rate_hertz=24000,
                                            language_code="en-US"
                                        )
                                        response = client.recognize(config=config, audio=audio)
                                        for result in response.results:
                                            transcript = result.alternatives[0].transcript
                                            await client_websocket.send(json.dumps({"audioText": transcript}))
                                        accumulated_audio = bytearray()
                                    except Exception as e:
                                        
                                        await client_websocket.send(json.dumps({
                                            "error": f"Transcription error: {str(e)}",
                                            "status": "error"
                                        }))
                                await client_websocket.send(json.dumps({"text": accumulated_responses}))
                except websockets.exceptions.ConnectionClosedOK:
                    print("Client connection closed normally (receive)")
                except Exception as e:
                    await client_websocket.send(json.dumps({
                        "error": str(e),
                        "status": "error"
                    }))
                finally:
                    print("Gemini connection closed (receive)")

            send_task = asyncio.create_task(send_to_gemini())
            receive_task = asyncio.create_task(receive_from_gemini())
            await asyncio.gather(send_task, receive_task)

    except Exception as e:
        await client_websocket.send(json.dumps({
            "error": "Session limit reached or connection error",
            "status": "session_closed"
        }))
    finally:
        print("Gemini session closed.")





######################################  Function To Restart the process ##############################################


async def reset_connections():
    try:
        # Cancel all existing tasks except the current one
        current_task = asyncio.current_task()
        for task in asyncio.all_tasks():
            if task is not current_task:
                task.cancel()
        
        # Wait for tasks to be cancelled
        await asyncio.sleep(1)
        
        # Reset Gemini client
        global client
        client = genai.Client(
            http_options={
                'api_version': 'v1alpha',
            }
        )
        
        # Start new webhook server
        async with websockets.serve(gemini_session_handler, "localhost", 9083):
            print("Webhook server restarted successfully on localhost:9083")
            await asyncio.Future()
    except Exception as e:
        error_message = f"Reset failed: {str(e)}"
        print(error_message)
        # Wait before retry
        await asyncio.sleep(2)
        await reset_connections()
    finally:
        # Cleanup any remaining connections
        for task in asyncio.all_tasks():
            if task is not current_task:
                task.cancel()
        
        
######################################  Function To Restart the process ##############################################





### EDIT THIS IF NEEDED ###
### ONLY FOR CHANGING WEBSOCKET CONNECTION DETAILS ###
        
async def main() -> None:
    async with websockets.serve(gemini_session_handler, "localhost", 9083):
        print("Running websocket server ")
        await asyncio.Future()  # Keep the server running indefinitely


if __name__ == "__main__":
    asyncio.run(main())