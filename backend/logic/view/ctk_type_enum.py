from enum import Enum

class CtkTypeEnum(Enum):
  """ Enum class to define different types of messages and their colors. """
  ERROR = {"type": "error", "color": "red"}
  WARNING = {"type": "warning", "color": "yellow"}
  OK = {"type": "ok", "color": "green"}