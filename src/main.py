from fastapi import FastAPI
from pydantic import BaseModel
from src.agent import LocalAgent
from src.scraping.url_processor import UrlProcessor
from src.config import Config
from fastapi.responses import StreamingResponse

app = FastAPI()
agent = LocalAgent()

class AskRequest(BaseModel):
    prompt: str
    stream: bool = False

@app.on_event("startup")
async def startup_event():
    url_processor = UrlProcessor(agent.web_scraper, agent.model.get_session_history(agent.session_id))
    url_processor.process_urls_from_file(Config.URLS_FILE)

@app.on_event("shutdown")
async def shutdown_event():
    agent.close()

@app.post("/ask")
async def ask(request: AskRequest):
    if request.stream:
        return StreamingResponse(agent.ask(request.prompt, stream=True), media_type="text/event-stream")
    else:
        response = agent.ask(request.prompt, stream=False)
        return {"response": response}
