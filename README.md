# Gemini Websocket Server

This Python application implements a WebSocket server that interfaces with Google's Gemini AI API, 
enabling real-time communication between clients and the Gemini model for screen sharing assistance.

## Prerequisites
**Python** > 3.9

**Required packages**: websockets google-genai python-dotenv

```
pip install websockets google-genai python-dotenv
```
## Config 

1. Modify  .env file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
GOOGLE_APPLICATION_CREDENTIALS="<PATH TO CREDENTIALS>"
```
2. The server runs on:
 ```
   host: localhost
   port: 9083
 ```
## Error Handling 

The server includes automatic restart capabilities and comprehensive error handling for:
1. Session limits
2. API communication failures
