from collections.abc import Sequence
from typing import Any

from agent_hub.domain import (
    Decision,
    Delivered,
    FileDelivery,
    OutgoingFile,
    Question,
    QuestionOption,
    QuestionsOutcome,
    ToolRequest,
)

DELIVERED = Delivered()

ASK_INPUT: dict[str, Any] = {
    "questions": [
        {
            "question": "Какой формат?",
            "header": "Формат",
            "options": [
                {"label": "Кратко", "description": "Только суть"},
                {"label": "Подробно", "description": "С примерами"},
            ],
            "multiSelect": False,
        }
    ]
}
ASK_QUESTION = Question(
    "Какой формат?",
    "Формат",
    (QuestionOption("Кратко", "Только суть"), QuestionOption("Подробно", "С примерами")),
    multi_select=False,
)


class FakeChannel:
    def __init__(
        self,
        decision: Decision,
        outcome: QuestionsOutcome,
        delivery: FileDelivery = DELIVERED,
    ) -> None:
        self.decision = decision
        self.outcome = outcome
        self.delivery = delivery
        self.requests: list[ToolRequest] = []
        self.asked: list[Sequence[Question]] = []
        self.sent: list[OutgoingFile] = []

    async def send_file(self, file: OutgoingFile) -> FileDelivery:
        self.sent.append(file)
        return self.delivery

    async def request(self, tool: ToolRequest) -> Decision:
        self.requests.append(tool)
        return self.decision

    async def ask(self, questions: Sequence[Question]) -> QuestionsOutcome:
        self.asked.append(questions)
        return self.outcome
