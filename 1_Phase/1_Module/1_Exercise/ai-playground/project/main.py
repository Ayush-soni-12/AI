
import json
from google import genai
from google.genai import types, errors
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = genai.Client()


class ChatRequest(BaseModel):
    prompt:str
    model:str
    temperature:float = 0.5
    # max_output_token:int = 200


class TopicAnalysis(BaseModel):
    topic: str
    difficulty: str
    concepts: list[str]


@app.get("/")
def home():
    return {"message": "AI Playground API is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = client.models.generate_content(
            model=request.model,
            contents=request.prompt,
            config=types.GenerateContentConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_output_token,
            ),
        )

        return {
            "response": response.text,
            "usage": response.usage_metadata,
        }
    except errors.APIError as e:
        raise HTTPException(status_code=e.code or 500, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):

    def generate():

        try:
            response = client.models.generate_content_stream(
                model=request.model,
                contents=request.prompt,
                config=types.GenerateContentConfig(
                    temperature=request.temperature,
                    # max_output_tokens=request.max_output_token,
                ),
            )

            for chunk in response:

                if chunk.text:
                    event = {
                        "type": "token",
                        "content": chunk.text
                    }

                    yield f"data: {json.dumps(event)}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:

            event = {
                "type": "error",
                "message": str(e)
            }

            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.post("/chat/structured")
def structured_output(request: ChatRequest):
    try:
        response = client.models.generate_content(
            model=request.model,
            contents=request.prompt,
            config=types.GenerateContentConfig(
                temperature=request.temperature,
                # max_output_tokens=request.max_output_token,
                response_mime_type="application/json",
                response_schema=TopicAnalysis,
            ),
        )

        # response.parsed contains the instantiated TopicAnalysis Pydantic model
        print(response)
        return {
            "response": response.parsed
        }
    except errors.APIError as e:
        # Gracefully handle API errors like 503 high demand or quota limits
        raise HTTPException(status_code=e.code or 500, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))