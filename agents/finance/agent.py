from core.src.base_agent import BaseAgent
from core.src.model.llm_model_graph import LlmModelGraph

from .config import FinanceConfig


class FinanceAgent(BaseAgent):
    """Financial Q&A agent.  Inherits all pipeline capabilities from BaseAgent;
    overrides only the system prompt."""

    def get_system_prompt(self) -> str:
        return LlmModelGraph._load_file(FinanceConfig.FINANCE_PROMPT_FILE)


if __name__ == "__main__":
    agent = FinanceAgent()
    with agent:
        try:
            while True:
                q = input("You: ")
                if q.lower() in ["exit", "quit"]:
                    break
                print(agent.ask(q))
        except KeyboardInterrupt:
            pass
