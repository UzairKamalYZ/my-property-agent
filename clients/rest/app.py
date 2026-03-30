from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
import asyncio

from agentP.src.agent import LocalAgent
from agentP.src.config.config import Config

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Validates the API key header when API_KEY is configured."""
    expected = Config.API_KEY
    if expected and api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent = LocalAgent()
    app.state.agent = agent
    print("✅ Agent started.")
    yield
    agent.close()
    print("🧹 Agent closed successfully.")


app = FastAPI(lifespan=lifespan)


async def sse_formatter(stream):
    """Formats a generator of strings into Server-Sent Events."""
    try:
        for chunk in stream:
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        print("Client disconnected.")


@app.get("/ask")
async def ask(prompt: str, stream: bool = False, _: None = Depends(verify_api_key)):
    """Endpoint to ask the agent a question."""
    agent = app.state.agent
    if stream:
        response_stream = agent.ask(prompt, stream=True)
        return StreamingResponse(sse_formatter(response_stream), media_type="text/event-stream")
    else:
        response = agent.ask(prompt, stream=False)
        return {"response": response}
