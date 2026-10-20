import json
import uuid
from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage


class EpisodeMessages(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("9c449eb4-0a7b-4d32-87ad-6bbb0b236251"))
        self.messages: list[dict] = []

    def on_message_received(self, msg: IncomingMessage) -> None:
        self.messages.append(json.loads(msg.read_string()))
