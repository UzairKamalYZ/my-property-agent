import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
import asyncio

from agentP.src.agent import LocalAgent
from agentP.src.config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("REST API starting up — initializing LocalAgent")
    agent = LocalAgent()
    app.state.agent = agent
    logger.info("REST API ready — agent initialized successfully")

    yield

    # Shutdown
    logger.info("REST API shutting down — closing agent")
    agent.close()
    logger.info("REST API shutdown complete")


app = FastAPI(lifespan=lifespan)


async def sse_formatter(stream, prompt: str):
    """Formats a generator of strings into Server-Sent Events."""
    chunk_count = 0
    try:
        for chunk in stream:
            chunk_count += 1
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.01)
        logger.debug("SSE stream complete: %d chunks sent prompt_len=%d",
                     chunk_count, len(prompt))
    except asyncio.CancelledError:
        logger.info("SSE stream cancelled — client disconnected after %d chunks", chunk_count)


@app.get("/ask")
async def ask(prompt: str, stream: bool = False):
    """Endpoint to ask the agent a question."""
    logger.info("GET /ask prompt_len=%d stream=%s", len(prompt), stream)
    agent = app.state.agent
    if stream:
        logger.debug("Starting SSE stream for prompt_len=%d", len(prompt))
        response_stream = agent.ask(prompt, stream=True)
        return StreamingResponse(
            sse_formatter(response_stream, prompt),
            media_type="text/event-stream",
        )
    else:
        response = agent.ask(prompt, stream=False)
        logger.info("GET /ask response ready prompt_len=%d response_len=%d",
                    len(prompt), len(str(response)))
        return {"response": response}
