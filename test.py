from main import FAST_MODEL, query_model, FakeTestErrorException, webhook
from os import getenv
import pytest

import telegram
import telegram.ext

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

example_image_request_json = {
    "update_id": 000000000,
    "message": {
        "message_id": 0000,
        "chat": {
            "id": ADMIN_CHAT_ID,
            "first_name": "no",
            "type": "private",
            },
        "date": 0000000000,
        "from": {
            "first_name": "no",
            "id": ADMIN_CHAT_ID,
            "is_bot": False,
            },
        "photo": [
            {
                "file_id": "blahblahblah",
                "file_unique_id": "blah",
                "file_size": 1920,
                "width": 90,
                "height": 80,
            },
        ]
    }  
}

example_document_request_json = {
    "update_id": 000000000,
    "message": {
        "message_id": 0000,
        "chat": {
            "id": ADMIN_CHAT_ID,
            "first_name": "no",
            "type": "private",
            },
        "date": 0000000000,
        "from": {
            "first_name": "no",
            "id": ADMIN_CHAT_ID,
            "is_bot": False,
            },
        "document": {
            "file_id": "blahblahblah",
            "file_unique_id": "blah",
            "file_name": "test.txt",
            "mime_type": "text/plain",
            "file_size": 1920,
        }
    }  
}

example_query = [{"role": "system", "content": "say 'h'"}]

@pytest.fixture
def mock_query_model(monkeypatch):
    def mock_query_model(previous_messages, model="MOCK_MODEL"):
        return "MOCK RESPONSE"
    monkeypatch.setattr("main.query_model", mock_query_model)

@pytest.fixture
def mock_message_user(monkeypatch):
    async def mock_message_user(chat_id, msg, context, parse_mode=None):
        print(f"mock_message_user called with {chat_id}, {msg}, {context}, {parse_mode}")
    monkeypatch.setattr("main.message_user", mock_message_user)

@pytest.fixture
def mock_message_user_gen(monkeypatch):
    def _apply_mock(key_message):
        async def mock_message_user_(chat_id, msg, context, parse_mode=None):
            if msg == key_message:
                raise FakeTestErrorException("received fake message")
            print(f"mock_message_user called with {chat_id}, {msg}, {context}, {parse_mode}")
        monkeypatch.setattr("main.message_user", mock_message_user_)
    return _apply_mock

@pytest.fixture
def mock_unsupported_message(monkeypatch):
    async def mock_unsupported_message(update, context):
        raise FakeTestErrorException("unsupported_message")
    monkeypatch.setattr("main.unsupported_message", mock_unsupported_message)

@pytest.fixture
def mock_read_document(monkeypatch):
    async def mock_read_document(message):
        return "FAKE DOCUMENT"
    monkeypatch.setattr("main.read_document", mock_read_document)

@pytest.mark.usefixtures("mock_query_model")
def test_webhook():
    """Test that the webhook returns a 200 status code. Doesn't send a real query to OpenAI, but does send a real response to the admin."""
    request = MockRequest(example_request_json)

    _, success = webhook(request)
    assert success == 200

def test_query_model():
    """Run a query on gpt-3.5-turbo and make sure it returns a string."""
    assert isinstance(query_model(example_query, FAST_MODEL), str)

# mock_message_user shouldn't be used, but is here to prevent spamming in case of errors
@pytest.mark.usefixtures("mock_query_model", "mock_message_user", "mock_unsupported_message")
def test_webhook_image():
    request = MockRequest(example_image_request_json)
    # check that the unsupported message function is called

    _, success = webhook(request)
    assert success == FakeTestErrorException.code()

@pytest.mark.usefixtures("mock_query_model", "mock_read_document")
def test_webhook_document(mock_message_user_gen):
    mock_message_user_gen("document received")
    request = MockRequest(example_document_request_json)

    _, success = webhook(request)
    assert success == FakeTestErrorException.code()
    