import pandas as pd

from .event_detection_average_precision import score


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    # Events with a missing step mark nights without that event. Counted as positives they
    # capped recall at the share of events with a step (0.730 on the TimeSeriesGym split, below
    # the Kaggle gold threshold); the competition's own solution contains only events with a step.
    answers = answers.dropna(subset=["step"])
    return score(
        solution=answers,
        submission=submission,
        tolerances={
            "onset": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
            "wakeup": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
        },
        series_id_column_name="series_id",
        time_column_name="step",
        event_column_name="event",
        score_column_name="score",
        use_scoring_intervals=False,
    )
