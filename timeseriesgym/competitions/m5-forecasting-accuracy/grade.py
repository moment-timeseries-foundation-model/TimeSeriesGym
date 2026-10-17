import numpy as np
import pandas as pd
from tqdm import tqdm

from timeseriesgym.utils import InvalidSubmissionError


## evaluation metric
## from https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834 and edited to get
# scores at all levels
class WRMSSEEvaluator:
    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        calendar: pd.DataFrame,
        prices: pd.DataFrame,
    ):
        train_y = train_df.loc[:, train_df.columns.str.startswith("d_")]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df["all_id"] = 0  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith("d_")].columns.tolist()
        valid_target_columns = valid_df.loc[
            :, valid_df.columns.str.startswith("d_")
        ].columns.tolist()

        if not all(c in valid_df.columns for c in id_columns):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()

        self.group_ids = (
            "all_id",
            "cat_id",
            "state_id",
            "dept_id",
            "store_id",
            "item_id",
            ["state_id", "cat_id"],
            ["state_id", "dept_id"],
            ["store_id", "cat_id"],
            ["store_id", "dept_id"],
            ["item_id", "state_id"],
            ["item_id", "store_id"],
        )

        for i, group_id in enumerate(tqdm(self.group_ids)):
            train_y = train_df.groupby(group_id)[train_target_columns].sum()
            scale = []
            for values in train_y.to_numpy():
                series = values[np.argmax(values != 0) :]
                scale.append(((series[1:] - series[:-1]) ** 2).mean())
            setattr(self, f"lv{i + 1}_scale", np.array(scale))
            setattr(self, f"lv{i + 1}_train_df", train_y)
            setattr(
                self, f"lv{i + 1}_valid_df", valid_df.groupby(group_id)[valid_target_columns].sum()
            )

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f"lv{i + 1}_weight", lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index("d")["wm_yr_wk"].to_dict()
        weight_df = self.train_df[["item_id", "store_id", *self.weight_columns]].set_index(
            ["item_id", "store_id"]
        )
        weight_df = weight_df.stack().reset_index().rename(columns={"level_2": "d", 0: "value"})
        weight_df["wm_yr_wk"] = weight_df["d"].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how="left", on=["item_id", "store_id", "wm_yr_wk"])
        weight_df["value"] = weight_df["value"] * weight_df["sell_price"]
        weight_df = weight_df.set_index(["item_id", "store_id", "d"]).unstack(level=2)["value"]
        weight_df = weight_df.loc[
            zip(self.train_df.item_id, self.train_df.store_id, strict=False), :
        ].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f"lv{lv}_valid_df")
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f"lv{lv}_scale")
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: pd.DataFrame | np.ndarray):
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        group_ids = []
        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(
                valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1
            )
            weight = getattr(self, f"lv{i + 1}_weight")
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            group_ids.append(group_id)
            all_scores.append(lv_scores.sum())

        return group_ids, all_scores


def grade(submission: pd.DataFrame, answers: list) -> float:
    """
    modified from
    https://www.kaggle.com/code/rohanrao/m5-how-to-get-your-public-lb-score-rank#Validation-Data
    """
    source_records = {}
    for record in answers:
        # Assume each record has a "source" key.
        src = record.get("source")
        if src not in source_records:
            source_records[src] = []
        source_records[src].append(record)

    # Now create separate DataFrames for each source without merging columns from different CSVs.
    dfs = {
        src: pd.DataFrame(records).drop(columns=["source"], errors="ignore")
        for src, records in source_records.items()
    }

    sell_prices = dfs["sell_prices"]
    calendar = dfs["calendar"]
    entire_df = dfs["test"]
    df_sample_submission = dfs["sample_submission"]
    df_sample_submission["order"] = range(df_sample_submission.shape[0])

    df_train = entire_df.iloc[:, :-28]
    df_valid = entire_df.iloc[:, -28:]

    # pre-process submission: the inner merge below would silently drop unknown ids and keep
    # repeated ones, shifting rows onto other series
    if "id" not in submission.columns:
        raise InvalidSubmissionError("Submission must have an `id` column")
    if submission["id"].duplicated().any() or set(submission["id"]) != set(
        df_sample_submission["id"]
    ):
        raise InvalidSubmissionError("Submission ids must be the sample's ids, each once")
    submission = (
        submission.merge(df_sample_submission[["id", "order"]], on="id")
        .sort_values("order")
        .drop(["id", "order"], axis=1)
        .reset_index(drop=True)
    )
    # forecast columns F1..F28 are days d_1914..d_1941
    submission.rename(columns={f"F{i}": f"d_{1913 + i}" for i in range(1, 29)}, inplace=True)

    evaluator = WRMSSEEvaluator(df_train, df_valid, calendar, sell_prices)
    group_ids, scores = evaluator.score(submission)

    return np.mean(scores)
