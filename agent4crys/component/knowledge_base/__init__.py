from .periodic_table import PTRetriever
from .database import ReasonRetriever


def load_knowledge_base():
    periodic_table = PTRetriever()
    reason_knowledge_base = ReasonRetriever()
    return (periodic_table, reason_knowledge_base)
