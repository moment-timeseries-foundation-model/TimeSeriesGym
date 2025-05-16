from collections import defaultdict


def grade(submission: defaultdict, answers: defaultdict) -> float:
    return int(submission == answers)
