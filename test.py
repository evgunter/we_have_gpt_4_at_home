from main import FAST_MODEL, query_model, setup_app_polling, webhook
from os import getenv
import pytest

class MockRequest():
    def __init__(self, json):
        self.json = json
        self.method = "POST"

    def get_json(self, force=False):
        return self.json

ADMIN_CHAT_ID = getenv("ADMIN_CHAT_ID")  # so it doesn't complain about being unverified

example_request_json = {
    "update_id": 000000000,
    "message": {
        "message_id": 0000,
        "from": {
            "id": ADMIN_CHAT_ID,
            "is_bot": False,
            "first_name": "no",
        },
        "chat": {
            "id": ADMIN_CHAT_ID,
            "first_name": "no",
            "type": "private",
        },
        "date": 0000000000,
        "text": "test request",
    } 
}

example_query = [{"role": "system", "content": "say 'h'"}]

@pytest.fixture
def mock_query_model(monkeypatch):
    def mock_query_model(previous_messages, model="MOCK_MODEL"):
        return "MOCK RESPONSE"
    monkeypatch.setattr("main.query_model", mock_query_model)

@pytest.mark.usefixtures("mock_query_model")
def test_webhook():
    """Test that the webhook returns a 200 status code. Doesn't send a real query to OpenAI, but does send a real response to the admin."""
    request = MockRequest(example_request_json)

    _, success = webhook(request)
    assert success == 200

def test_query_model():
    """Run a query on gpt-3.5-turbo and make sure it returns a string."""
    assert isinstance(query_model(example_query, FAST_MODEL), str)
