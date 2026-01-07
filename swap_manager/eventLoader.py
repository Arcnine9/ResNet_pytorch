import re
from typing import List, Dict

class Event:
    def __init__(self, issued_time: int, tensor_id: int, from_location: str, to_location: str, tag: str):
        self.issued_time = issued_time
        self.tensor_id = tensor_id
        self.from_location = from_location
        self.to_location = to_location
        self.tag = tag

    def __repr__(self):
        return (f"Issued Time: {self.issued_time}, Tensor: {self.tensor_id}, "
                f"From: {self.from_location}, To: {self.to_location}, Tag: {self.tag}")


class eventLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.events: List[Event] = []

    def load_events(self) -> List[Event]:
        """从文件中加载事件列表"""
        pattern = re.compile(r"Issued Time: (\d+) Tensor: (\d+) From: (\w+), To: (\w+) tag: (\w+)")
        with open(self.file_path, 'r') as file:
            for line in file:
                match = pattern.match(line.strip())
                if match:
                    issued_time = int(match.group(1))
                    tensor_id = int(match.group(2))
                    from_location = match.group(3)
                    to_location = match.group(4)
                    tag = match.group(5)
                    event = Event(issued_time, tensor_id, from_location, to_location, tag)
                    self.events.append(event)
        return self.events

    def get_events(self) -> List[Event]:
        """获取加载的事件列表"""
        return self.events


# 示例用法
if __name__ == "__main__":
    event_loader = EventLoader("events.txt")  # 假设事件列表文件名为 events.txt
    events = event_loader.load_events()
    for event in events:
        print(event)