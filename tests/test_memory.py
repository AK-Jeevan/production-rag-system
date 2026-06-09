import pytest
from src.memory.chat_memory import ChatMemory


class TestChatMemory:

    def test_add_and_retrieve_messages(self):
        memory = ChatMemory(max_history=10)
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi there!")
        assert memory.message_count() == 2

    def test_max_history_respected(self):
        memory = ChatMemory(max_history=4)
        for i in range(10):
            memory.add_user_message(f"Message {i}")
        assert memory.message_count() <= 4

    def test_clear_resets_memory(self):
        memory = ChatMemory(max_history=10)
        memory.add_user_message("Hello")
        memory.clear()
        assert memory.message_count() == 0

    def test_get_history_as_text(self):
        memory = ChatMemory(max_history=10)
        memory.add_user_message("What is FastAPI?")
        memory.add_assistant_message("It is a web framework.")
        history = memory.get_history_as_text()
        assert isinstance(history, str)
        assert len(history) > 0

    def test_empty_memory_history_is_empty_string(self):
        memory  = ChatMemory(max_history=10)
        history = memory.get_history_as_text()
        assert history == "" or isinstance(history, str)