from .questions import Choice, Score, Noul, Query


def __getattr__(name):
    if name == "SystemOneVL":
        from .model import SystemOneVL
        return SystemOneVL
    raise AttributeError(name)
