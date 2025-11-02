from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from src.agent import LocalAgent
from src.scraping.url_processor import UrlProcessor
from src.config import Config
from pydantic import BaseModel
import asyncio

class AskRequest(BaseModel):
    prompt: str
    stream: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    agent = LocalAgent()
    app.state.agent = agent  # store agent in app state
    url_processor = UrlProcessor(agent.web_scraper, agent.model.get_session_history("default"))
    url_processor.process_urls_from_file(Config.URLS_FILE)
    print("✅ Agent started and URLs processed.")

    yield  # <--- everything before this runs at startup, after runs at shutdown

    # Shutdown
    agent.close()
    print("🧹 Agent closed successfully.")

app = FastAPI(lifespan=lifespan)

async def sse_formatter(stream):
    loop = asyncio.get_event_loop()
    q = asyncio.Queue()

    def run_in_thread():
        for chunk in stream:
            loop.call_soon_threadsafe(q.put_nowait, chunk)
        loop.call_soon_threadsafe(q.put_nowait, None) # Signal end of stream

    await asyncio.to_thread(run_in_thread)

    while True:
        chunk = await q.get()
        if chunk is None:
            break
        yield f"data: {chunk}\n\n"
        await asyncio.sleep(0.01) # Add a small delay

@app.get("/ask")
async def ask(prompt: str, stream: bool = False):
    agent = app.state.agent  # retrieve from app state
    if stream:
        stream = agent.ask(prompt, stream=True)
        return StreamingResponse(sse_formatter(stream), media_type="text/event-stream")
    else:
        response = agent.ask(prompt, stream=False)
        return {"response": response}
