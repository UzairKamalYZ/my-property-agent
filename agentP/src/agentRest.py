from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from agentP.src.agent import LocalAgent
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    agent = LocalAgent()
    app.state.agent = agent  # store agent in app state
    print("✅ Agent started.")

    yield  # <--- everything before this runs at startup, after runs at shutdown

    # Shutdown
    agent.close()
    print("🧹 Agent closed successfully.")

app = FastAPI(lifespan=lifespan)

async def sse_formatter(stream):
    """Formats a generator of strings into Server-Sent Events."""
    try:
        for chunk in stream:
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.01)  # Small delay to allow client to process
    except asyncio.CancelledError:
        # This is expected if the client disconnects
        print("Client disconnected.")

@app.get("/ask")
async def ask(prompt: str, stream: bool = False):
    """Endpoint to ask the agent a question."""
    agent = app.state.agent
    if stream:
        response_stream = agent.ask(prompt, stream=True)
        return StreamingResponse(sse_formatter(response_stream), media_type="text/event-stream")
    else:
        response = agent.ask(prompt, stream=False)
        return {"response": response}
