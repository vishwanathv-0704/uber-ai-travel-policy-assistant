class ConversationMemory:
    """
    Simple short-term conversation memory.

    Stores user and assistant messages so that
    follow-up questions can use previous context.
    """

    def __init__(self, max_messages=10):
        self.messages = []
        self.max_messages = max_messages

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })
        self._trim()

    def add_assistant_message(self, message):
        self.messages.append({
            "role": "assistant",
            "content": message
        })
        self._trim()

    def get_history(self):
        return self.messages.copy()

    def get_context(self):
        """
        Convert conversation history into text
        that can be supplied to the LLM.
        """

        if not self.messages:
            return ""

        context = []

        for message in self.messages:
            role = message["role"].upper()
            content = message["content"]

            context.append(
                f"{role}: {content}"
            )

        return "\n".join(context)

    def clear(self):
        self.messages = []

    def _trim(self):
        """
        Keep only the most recent messages.
        """

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
