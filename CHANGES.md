# Changes affecting scores, validity and behaviour

This file lists every change on this branch, relative to upstream `main` (520fea8), that changes
a score, whether a submission is valid, the data a challenge is prepared with, or how the tools
behave. Each entry gives the old and new behaviour, the reason, and the commit. Refactors, lint
and speed-ups that change no output are listed at the end.

Results produced before these changes are not directly comparable for the challenges named
under "Scores and validity", "Hyperparameter search", "Feature enhancement" and "Prepared data";
the medals are not comparable for those named under "Medals and leaderboards".

## Scores and validity

- **child-mind-institute-detect-sleep-states**: ground-truth events without a step are dropped
  (229df50). They counted as positives that can never be matched, so every score was too low and
  a perfect submission got mAP 0.730 (below the 0.825 gold threshold); it now gets 1.0.
- **RMSLE graders (online-sales, ashrae, sberbank, recruit)**: predictions must be finite and
  >= 0, and only cells without an answer are ignored (ad4b2c1). online-sales dropped every NaN
  term, so a prediction below -1 removed its cell from the score, and the other three returned
  NaN. Predictions in (-1, 0), which were scored before, are now invalid. Valid sample
  submissions score as before.
- **ashrae, recruit, sberbank, time-series-exam-1-1**: rows are matched on the declared id
  columns, not by position; missing, repeated or unknown ids are invalid (fd2e527). A correct
  submission in another row order was scored against the wrong rows. Correctly ordered
  submissions score bit-identically.
- **new-york-city-taxi-fare-prediction**: the sorted submission keys must equal the answer keys
  (fd2e527). Both sides were sorted but the keys never compared.
- **m5-forecasting-accuracy**: the submission must hold exactly the sample's 30,490 `_validation`
  ids, each once (692f473). The merge silently dropped unknown ids and kept repeated ones, so a
  repeated id shifted rows onto other series. Kaggle-format files that also hold the
  `_evaluation` rows, which the merge used to drop, are now invalid.
- **MIT-BIH-Arrhythmia**: the submission must have shape (N, 6) (2a79556). An (N, 1) or (N, 5)
  matrix was scored by an argmax over whatever columns it had.
- **MIT-BIH-Arrhythmia-Weak-Supervision**: classes are read by column name (`normal`, `pvc`) and
  the Id sets compared (2a79556). A file written as `Id,pvc,normal` had every label flipped.
- **amp, optiver, osic**: NaN or infinite predictions are invalid (aab930c). They made the score
  NaN, which was reported as a valid submission.
- **ucr-anomaly-detection**: the score averages over every dataset in the answers (229df50).
  It iterated over the submission's keys, so a submission with only the easiest dataset was
  scored on that one alone.
- **ucr-anomaly-detection**: the score is each dataset's average precision on the point labels
  (`sklearn.metrics.average_precision_score`), averaged over the datasets; it was the
  point-adjusted best F1 (d7cb84e). Point adjustment pays chance: on the shipped data, 20 seeded
  uniform-random submissions average 0.453 (0.377 to 0.543), and a label-free comb (every 400th
  point 1) earns 0.540, both above a rolling z-score (0.364). In average precision, a constant
  scores 0.0086 (the share of anomalous points), random scores 0.0095 (median of 20 seeds), the
  comb 0.0088, a nearest-neighbour distance to the training series 0.417, and the labels 1.0.
  The point-adjusted F1 is still logged. Scores must be finite numbers in the labels' shape: NaN
  and infinite scores were graded 0.0, and an (n, 1) column of the labels 1.0. The placeholder
  leaderboard stays ignored, so no medals.
- **gift-eval-short-horizon**: the score is GIFT-Eval's leaderboard metric, each dataset's MASE
  divided by the seasonal naive forecast's MASE, geometric mean over the 5 datasets (4c2294b).
  It was MAPE, which divides by max(|actual|, eps): a non-zero forecast where the actual value is
  exactly 0 gave 9.1e14 on hierarchical_sales, so the seasonal naive forecast scored 1.8e14 and
  the zero sample 0.9947. Lower is better: seasonal naive now scores 1.0, naive 1.02262, the zero
  sample 5.63105 and a perfect forecast 0; per dataset, the seasonal naive and naive MASE equal
  GIFT-Eval's published values to 1e-9. Infinite predictions are invalid, like NaN ones. MAPE is
  still logged. The answers must be prepared again (see "Prepared data").
- **moment-anomaly-detection-score, time-series-library-itransformer-weather-forecast**: the shape
  is checked before `np.isclose` (229df50). It broadcasts, so a (1, n) or scalar submission
  could match.
- **Loaders**: a missing submission, script or model file is an invalid submission (f42dee0).
  The loaders returned the exception instead of raising it: an HPS folder without models scored
  -inf as "same hyperparameters", and a missing feature-enhancement file crashed the grader.
- **Pickled submissions (context-is-key-moirai, context-is-key-nn5-chronos)**: decoded with an
  allowlist of plain containers, numbers and numpy scalars; anything else, numpy arrays included,
  is invalid (56dd429, d286794). `pickle.load` ran any code the file named, also as root in the
  grading server, and arrays let the file set how much memory is allocated. The shipped samples
  and answers decode unchanged.
- **HDF5 submissions**: soft and external links, virtual datasets, external storage, committed
  datatypes and link cycles are invalid (cdf27c1). h5py followed them, so a submission could
  have the private answers read and graded as its own predictions.
- **HDF5 submissions**: the datasets may declare at most 4 GiB in total, and when graded or
  validated at most 8 times the size of the answers file, but not less than 64 MiB (2726419).
  A 2.4 kB file declaring 3.2 GB of never-written data was read into memory and the grading
  process was OOM-killed. The largest legitimate submission (TSL long-horizon in float64)
  declares 3,684 MiB.
- **.npy submissions**: the header is read first, and the array may declare at most the HDF5 cap
  above and must store all the data it declares; a zip named `.npy` is invalid (d7a2382).
  `np.load` read all the declared data (a sparse file of 100 MiB of zeros, for MOMENT's 50
  answers, was read in full before the shape check), and returned a zip as an `NpzFile`. The
  shipped MOMENT and TSL iTransformer samples load as before.
- **resnet-tensorflow-to-pytorch**: the test model may be trained on at most
  `n_epochs * len(train_data)` samples (ec1faa1). The grader asked for `n_epochs=1` but never
  checked it: a Claude Code submission set `n_epochs = max(n_epochs, 30)` and passed. The grader
  now counts every sample back-propagated through an instance of the submission's `ResNet`
  (copies and reloaded models included) from before the submission is imported until the test
  model's predictions return, so copying the tensors out of the dataset does not hide extra
  passes; forward passes without back-propagation do not count. Above the budget the test
  fails. The accuracy test now runs before the structural checks, so nothing trained in the
  structural call of `train` carries over. The count runs in the same process as the
  submission, so code written to evade it is not stopped; it catches code that trains for more
  passes than asked, whatever loop or copy of the data it uses.
- **resnet-tensorflow-to-pytorch**: the training count is taken at the layers (7d81be5). The
  first counter hooked calls of the submission's `ResNet` only, so a port that trains through
  another method, as `model.logits(x)` in a Claude Code evaluation submission, counted 0
  samples, and an over-trained port using that idiom would have passed. Now every
  back-propagated batch through layers with parameters counts once, through any method, a
  submodule or `torch.func`. A test model more accurate than the best constant class with no
  counted training sample is an invalid submission.
- **resnet-tensorflow-to-pytorch**: the grading arguments ask for `n_epochs=40` instead of 1
  (f1ca4fc). A port with Keras' BatchNorm settings (PyTorch momentum 0.01) scored 0.49 to 0.51
  after 1 epoch on 10 of 10 seeds: its running statistics lag the weights and eval-mode
  predictions collapse to one class. Measured in tsgym-core on CPU, it passes on 12 of 20 seeds
  after 25 epochs, 30 of 31 after 30 and 31 of 31 after 40. Ports with PyTorch's default
  momentum pass from 3 epochs, and the reference port with re-estimated statistics passes at
  every count. An untrained model and a one-layer convolution with average pooling score 0.49
  to 0.51 at every count, so they still fail. A ResNet without its shortcuts also reached
  accuracy 1.0 after 30 epochs; the architecture check below now fails it. The threshold stays
  at 0.8, because the accuracies are either about 0.5 or about 1.0. Grading a faithful port
  takes about 70 s.
- **resnet-tensorflow-to-pytorch**: the test also checks the architecture (b402c61). A fresh
  `ResNet` built with the grading arguments must give a gradient, in one forward and backward
  pass on `x.pt`, to exactly as many trainable parameters as the Keras model of `resnet.py` has
  with the same arguments: 504,258 here, as Keras' model summary counts them. Before, the
  accuracy test alone accepted another model: on this data a linear model reaches accuracy 1.0,
  and so does a ResNet without its shortcuts after 30 epochs. Every honest port measured uses
  exactly 504,258: a port with Keras' settings, one with PyTorch's BatchNorm defaults, the
  reference port and two Claude Code ports. A ResNet without its shortcuts uses 495,170, also
  when it still defines the unused shortcut layers. The description states the rule and the
  number.
- **stomp-R-to-python**: each test also runs on a variant of its inputs, and passes only if the
  outputs match on both (d918a7c). The tests used only the public inputs, so outputs computed
  offline and pasted into the code passed. The variant reverses the inputs in time and maps
  them by `-1.5 * x + 0.5`. The z-normalized distance does not change under either, so the R
  answers give the variant's expected outputs exactly, and no reference implementation is
  added. The reference port and two Claude Code ports pass as before.
- **Unreadable submissions**: a ragged CSV, a wrong suffix, an unreadable `.h5` or `.npy`, or a
  code submission that does not import, is reported as invalid instead of aborting the whole
  `timeseriesgym grade` batch (5e8750c). One bad agent file used to lose the report for every
  other run.

## Hyperparameter search

Both challenges: optiver-realized-volatility-prediction-hyperparameter-search and
ptb-xl-classification-challenge-hyperparameter-search.

- **Score**: `sign * (improved - sample) / sample`, the relative improvement over the challenge's
  sample submission; it was the improvement over the agent's own `baseline_submission.csv`
  (28dd3b2). Nothing ties that file to the baseline model, so a sabotaged baseline raised the
  score (Optiver 0.72 to 0.998 with a baseline 100 times the targets). Optiver's score is now
  `1 - RMSPE` (an honest example: 0.721 before, 0.823 now); PTB-XL's maximum is about 7.97.
  The baseline files are still required and only decide validity.
- **Optiver baseline**: must be `xgboost.XGBRegressor()` with default hyperparameters; only
  `n_jobs`, `nthread`, `verbosity` and `random_state` may be set (d0d8e80). The description
  requires this baseline, but any baseline was accepted.
- **Solution code**: required (a script or a code directory) and checked before any submitted
  pickle is loaded; without it the submission is invalid (977abdd). The search requirement was
  checked only when a solution file was passed, which by default it was not.
- **What counts as a search**: the code must call a search API through its imports and, where
  that call only sets the search up, the method that runs it, for example `optuna.create_study`
  with `study.optimize`, or `GridSearchCV` with `fit` (7c12345; earlier 229df50, 7339bf6). A bare
  `import optuna` used to need to appear twice; then any reference into a search module counted,
  including `train_test_split` and `optuna.__version__`. The PTB-XL description now lists the
  accepted libraries; a hand-written search is not recognised.
- **Same hyperparameters**: models with `get_params()` are compared on it, NaN-aware, and
  `n_jobs`, `nthread`, `verbosity` and `random_state` are not compared (229df50, 9e5e756).
  XGBoost's default `missing=nan` made identical models compare as different, and a change of
  `n_jobs` alone counted as a different model.
- **TorchScript models (PTB-XL)**: compared by layer settings (stride, padding, kernel size,
  channel and feature counts, bias) and layer kind, not only parameter shapes (c65ac91). Every
  scripted layer has the type `RecursiveScriptModule`, so models differing only in stride,
  padding or activation scored -inf.
- **Unreadable code files**: a non-UTF-8 file or a symlink under the code directory is skipped
  with a warning; symlinks are not followed (d4da33a). One such file made the whole run "not
  graded".
- **Missing sample submission**: grading stops with a setup error (48537c5). Every HPS run in
  the batch was recorded as an invalid submission.
- **Ranking**: HPS reports are never ranked against a leaderboard, and -inf is invalid (229df50).
  The inherited RMSPE leaderboard counted -inf as a gold medal. Optiver HPS now says
  `awards_medals: false` (696cfe0).

## Feature enhancement

One challenge: ptb-xl-classification-challenge-feature-enhancement.

- **Grading is reachable**: `grade-feature-enhancement`, `grade_feature_enhancement()` and JSONL
  dispatch (617f82d). The config was dropped when loaded, so the challenge could not be graded.
- **Rubric**: every function must annotate its parameters and its return (`__init__` may leave
  out `-> None`), every class needs a docstring, every substantive function is judged, `Args:`
  and `Returns:` must appear in docstrings, comments do not count, and TensorBoard logs need an
  `events.out.tfevents.*` file (1ecfecd, b8026f5). One annotation or one documented class
  passed, commented-out code scored, and any file under `runs/` counted as a log. The shipped
  example gets the same sub-scores as before.
- **YAML config**: a `#` inside a quoted value is not a comment (b8026f5). `name: "run #1"` was
  cut at the `#`, losing any `model:` key after it.
- **Reports**: dict scores are written as JSON objects (1fac171); they were a Python repr string
  that `from_dict` could not read.
- **Score** (`overall_score`): the test accuracy if every rubric check passes, and 0 otherwise
  (e87c88d); it was the mean of the TensorBoard, code-quality and Hydra sub-scores and the
  accuracy. Three quarters of it came from keyword and AST checks: with the rubric passed, any
  predictions scored at least 0.75, and the most frequent training class for every ECG 0.888.
  The lint check passes only with no flake8 problems; the code-quality sub-score keeps its
  partial credit. The sub-scores and the accuracy are still reported, with `rubric_passed`, and
  the description lists what each check requires. The design is the RL conversion's (rl_envs
  DEVIATIONS G12).
- **Model from the config**: the Hydra check passes if `train.py` builds the model from the
  config's `model` section: it passes the section, or a value read from it, to a class or
  function of `model.py` or to `hydra.utils.instantiate`, or it passes an argument to a class or
  function of `model.py` that reads a parameter's `model` section (ef6a6bd). The check looked for
  the text `cfg.model` or `cfg['model']` in `train.py`: it rejected `ECGClassifier(config)` with
  a model that reads `config.model`, and accepted `print(dict(cfg.model))`, which an agent added
  in an evaluation run to pass it. The shipped example passes as before.
- **flake8**: a dependency (7.1.1), and a flake8 that cannot run is a grading error (eda11ed). It
  was neither declared nor installed in the tsgym-core image; a missing flake8 gave half the lint
  credit, so no code-quality score could reach 1.0, and a run that failed without a report
  counted as no problems. The shipped example's one flake8 problem (E203) is fixed (d754b5d).
  Grading stops with a setup error when flake8 is not installed (ac59df8), as for a missing HPS
  sample submission; every run in the batch was graded invalid.
- **flake8 options**: the lint check runs `flake8 --isolated --disable-noqa --extend-ignore=E203
  --max-line-length=100` (3cc2cce); it ran `flake8 --max-line-length=100`. A `# noqa` comment
  or a configuration file in flake8's working directory could switch checks off, and E203
  failed the slices that black and ruff write (`x[i : i + 1]`), the repository's own format.
- **Code checks and event files**: the TensorBoard checks (the import, `SummaryWriter(...)`,
  `.add_scalar(...)`, `.add_graph(...)`) and the Hydra checks (the import, `@hydra.main`,
  `OmegaConf` or `DictConfig`) look at train.py's syntax tree; they were regular expressions,
  so code in a string literal passed them (a train.py that held a training script as a string
  scored 0.8 on both). A TensorBoard log must be an event file that holds a logged scalar
  (CRC-checked TFRecord records, as SummaryWriter writes them); any file named
  `events.out.tfevents.*`, even an empty one, counted (a7fbd91). The checks are the rl_envs
  rubric's (F4-04). The shipped example with SummaryWriter's own event file passes every check.

## Medals and leaderboards

- **Placeholder leaderboards**: one-row "dummy" leaderboards are ignored, so these 9 challenges
  award no medals: context-is-key-moirai, context-is-key-nn5-chronos,
  moment-anomaly-detection-score, ptb-xl-classification-challenge and its HPS and FE variants,
  time-series-exam-1-1, time-series-library-itransformer-weather-forecast, ucr-anomaly-detection
  (229df50). With one row, every score >= 0 was gold, even for lower-is-better metrics, and
  run-group medal totals counted every valid submission on them as gold.
- **gift-eval-short-horizon**: `leaderboard.csv` lists the models of GIFT-Eval's own leaderboard
  (Hugging Face space Salesforce/GIFT-Eval, revision 9d731e061162d5d8440204a2c455583d82e084d4 of
  2026-10-09), each scored with the new metric from its published MASE on the 5 datasets
  (b091026, dc7c088). It was a one-row placeholder (dummy, 0, on the MAPE scale), so no medals.
  Kept: models marked `"testdata_leakage": "No"` with a result on all 5 datasets, 123 of 143;
  the 20 others are flagged for leakage. Gold is rank 10 (0.71687), silver 24 (0.74775), bronze
  49 (0.76343), median 0.77337; Seasonal_Naive scores 1.0 and Naive 1.02262. `leaderboard.py`
  rebuilds the file from the pinned revision.
- **Non-finite scores** never earn a medal (229df50).
- **`is_lower_better` in reports** comes from each config's new `grader.lower_is_better`
  (b94eb3b). It came from the leaderboard's order only, so without a real leaderboard it was None
  and was written as false: wrong for csdi-implement-model-pm25, csdi-reproduce-pm25 and
  time-series-library-long-horizon-forecast, where lower is better. A direction that is still
  unknown (a config without the key) is written as null. Medals are ranked as before: a test
  checks that every configured direction agrees with the challenge's real leaderboard.
- **`awards_medals`** is now loaded into `Competition` but stays informational, as on Kaggle:
  new-york-city-taxi-fare-prediction and spaceship-titanic (`false`, real leaderboards) keep
  their medals (696cfe0).
- **osic-pulmonary-fibrosis-progression**: `leaderboard.csv` lists all 2,097 teams of Kaggle's
  final leaderboard (7b15e32); it listed the top 20, so Kaggle's rule ranked on 20 teams: gold at
  rank 2 (-6.8311), silver 4 (-6.8347), bronze 8 (-6.8374), median -6.8395. Now gold is rank 14
  (-6.8412), silver 104 (-6.8533), bronze 209 (-6.8683), median -6.8993. Scores do not change.
- **recruit-restaurant-visitor-forecasting**: `leaderboard.csv` is Recruit's final standings,
  2,150 rows (cbd8e31). It was another competition's leaderboard: 1,964 rows dated 2016-05-24 to
  2016-08-30, before Recruit's data ends (2017-05-31), with none of Recruit's teams. Gold 0.43593
  (rank 13) -> 0.51023 (rank 14), silver 0.44570 (98) -> 0.51810 (107), bronze 0.45034 (196) ->
  0.51817 (215), median 0.48947 -> 0.53223.
- **online-sales, sberbank-russian-housing-market, ashrae-energy-prediction**: `leaderboard.csv`
  is the final standings, not Kaggle's public leaderboard (aab3d79); the teams, row counts and
  medal ranks are the same. online-sales: gold 0.58432 -> 0.56957, silver 0.61887 -> 0.60111,
  bronze 0.66117 -> 0.64400, median 0.73573 -> 0.71913. sberbank: gold 0.30843 -> 0.31227,
  silver 0.31026 -> 0.31612, bronze 0.31035 -> 0.31649, median 0.32089 -> 0.32323. ashrae:
  gold 0.939 -> 1.244, silver 0.955 -> 1.268, bronze 0.978 -> 1.280, median 1.108 -> 1.352.
- **Leaderboard download** (`dev download-leaderboard --force`, `prepare --overwrite-leaderboard`):
  refuses to write a leaderboard with fewer rows than 95% of Kaggle's team count (c85f491). The
  view is paged and kaggle 1.6.17 cannot ask for more than its first page (20 rows), which is how
  OSIC's file was truncated; with that client, every competition of more than 21 teams is refused.
- **Shipped leaderboards** are tested to be in the leaderboard view's format (the final
  standings, not the download's public leaderboard), ranked, with unique teams, and, where the
  data have dates, dated after the data (556b87b).

## Prepared data and checksums

Rebuilding these challenges gives different files; results on the old data are not comparable.

- **amp-parkinsons-disease-progression-prediction**: one answer per `prediction_id`, joined on
  visit and UPDRS part (21a2a7e). Each id was paired with every part rated at the visit (11,004
  rows for 3,183 ids), so a perfect forecast scored SMAPE+1 72.97; it now scores 0. The sample
  submission scores 124.93847 (was 129.7763). Only `ground_truth.csv` and
  `sample_submission.csv` change.
- **new-york-city-taxi-fare-prediction**: the test split is cleaned as Kaggle's test set was
  (2bb750e). Rides with a pickup or dropoff outside the ranges of Kaggle's own `test.csv`
  (longitude -74.263242 to -72.986532, latitude 40.568973 to 41.709555) or with 0 or more than
  6 passengers are dropped from the same random sample. The test now has 9,638 rows (was
  9,914): 233 lay outside the area (219 with a zero coordinate) and 45 had 0 passengers, 2 of
  them both. The old test was a random sample of the raw training file, but the leaderboard
  (gold 2.83377) was measured on Kaggle's clean test, and these rows dominated the error: two
  agent submissions score RMSE 5.45 and 5.25 on the old test and 3.49 and 3.48 on the new one.
  The sample submission scores 9.93913 (was 10.02927). `labels.csv` is unchanged; `test.csv`,
  `sample_submission.csv` and the answers change, and the descriptions state the cleaning.
- **ptb-xl-classification-challenge (and its HPS and FE variants)**: training uses PTB-XL folds
  1-8 (12,957 records; was folds 1-7, 11,320 records) (e5d3422). `train_fold` was
  `np.arange(0, 8)`, but the folds are numbered 1-10, so fold 8 was in no split.
- **PTB-XL build**: records are in sorted (ecg_id) order, the h5 files are written without
  timestamps, `download_data.py --release_dir` builds offline, and the checksums of all three
  challenges are regenerated (045eb7a, decfd44). The order came from `os.listdir`, so builds on
  different hosts differed. Upstream's previous `test_label_df.csv` (most likely another row
  order) could not be reproduced.
- **Zips built by `utils.compress`**: entries are sorted and get a fixed date and mode (19ae628),
  so a rebuild is byte-identical. This changes the zip bytes of MIT-BIH-Arrhythmia and
  gift-eval-short-horizon (built by their `download_data.py`); their prepared files are
  unchanged.
- **MIT-BIH checksums**: MIT-BIH-Arrhythmia's zip and prepared h5 checksums, and the three data
  CSVs of MIT-BIH-Arrhythmia-Weak-Supervision, are regenerated from two identical rebuilds of the
  PhysioNet 1.0.0 release (129bd2e, 80294d9). The old zip checksum held file times and could never
  match; the old h5 values could not be reproduced with the pinned libraries (upstream's sample has
  4,141 test beats, a rebuild 4,139). The download now fails on HTTP errors (bf41cd7).
- **MIT-BIH-Arrhythmia-Weak-Supervision**: the Snorkel tutorial is downloaded from a commit
  (snorkel-tutorials 2ba84e4), not `refs/heads/master`, and the docs PDF from Release 0.9.7, not
  `latest`; both are checked against an MD5 checksum, also with `--skip-verification` (7789b14).
  The tutorial and the two extracted text files are unchanged; `snorkel-docs.pdf` differs from
  the `latest` build in 66 bytes of build metadata.
- **gift-eval-short-horizon**: the private `test.h5` also holds each test window's MASE scale
  and seasonal naive forecast, computed from the public test inputs, as the new score needs the
  windows' history: `labels` is now a group of `target`, `seasonal_error` and `seasonal_naive`,
  and its checksum is regenerated (4c2294b; two more prepares verify). The zip and the public
  files are unchanged. The documented source is the Git LFS bundle, whose zip matches
  `checksums.yaml`; the Hugging Face download path never matched it.
- **m5-forecasting-accuracy**: the public `sell_prices.csv` has every training week (6,597,201
  rows; was 4,088,360) (d64a36f). Days were compared as strings, so d_2..d_9 and d_200..d_999
  counted as later than d_1913. Scores do not change: the grader reads prices from the answers.
- **LANL-Earthquake-Prediction**: test segment ids come from a seeded generator, duplicates are
  rejected, and the checksums are regenerated (7a8862f). Every prepare named the 2,500 segments
  differently, so checksums could never verify.
- **LANL-Earthquake-Prediction**: the test is laid out as Kaggle's: 150,000-row segments from
  the held-out earthquake cycles 13-17, 1,068 in all, each labelled with `time_to_failure` at its
  last row (6cd897f, 3c3dcb0, 0ed1739, 1cee982). It was 500 random, possibly overlapping
  15,000-row windows from each of those cycles (2,500), which matched the training distribution
  so closely that a 4-minute LightGBM on summary statistics scored better than Kaggle's gold
  threshold (2.40159) and its rank-1 score (2.26589). Consecutive segments are a random
  4,096-8,191 rows apart, and each cycle's last segment ends a random 0-4,095 rows before the
  cycle's end (0ed1739). The signal comes in 4,096-row recording bins and every cycle ends on a
  bin boundary, so segments cut back to back from the cycle ends, as 6cd897f first did, gave away
  their distance to the earthquake in their bin phase: a predictor reading it with a table from
  `train.csv`, without modelling the signal, scored MAE 2.18. No segment spans an earthquake or
  comes within a bin of a training row. The gaps and the ids come from generators seeded with
  digests of private data, and the files, answers and sample are in id order (they were grouped
  by cycle), so neither ids nor order give a segment's position. `train.csv` (cycles 1-12) is
  byte-identical, and prepare reads it in chunks instead of all 629M rows at once. On the new
  test the sample scores 5.69511 (was 6.21077), the median of the training labels 3.00716, a
  LightGBM on 58 summary statistics of 150,000-row training windows 2.443 (2.409-2.443 over 5
  seeds, about 10 CPU minutes), and a stronger one (68 features, leave-one-cycle-out CV) 2.168,
  better than Kaggle's winner.
- **ashrae-energy-prediction**: the training labels end on 2016-08-31, two months before the
  test (031135d, ba50742). They ran to 2016-10-30, right up to the test (the last two months of
  2016), while Kaggle's test (2017-2018) lay up to two years after its training year. So a lookup
  table without any model, each building-meter's mean log reading over its last 7 training days,
  scored RMSLE 1.145, better than Kaggle's gold threshold (1.244) and its rank-1 score (1.231).
  The readings of 2016-09-01 to 2016-10-30 are now in neither split. The test, the sample, the
  answers, `building_metadata.csv` and both weather files are unchanged (the weather covers the
  whole year); only `train_label.csv` changes (13,358,457 rows; was 16,745,071), and its
  checksum is regenerated (two prepares, identical files). On the new split the same tables score
  1.852 (last 7 days), 1.840 (14) and 1.837 (28), the building-meter mean over all training days
  1.448, and a quick LightGBM 1.126 (about 5 CPU minutes, 3 seeds within 0.003), still better
  than gold. The sample still scores 4.7002. The description states the gap.
- **Cloned task repositories**: shipped without `.git` (1568f6d) and pinned to commits (8dbe2b8;
  MOMENT to 37a8bde in 00279f6). The history held the reference solution (iTransformer), and
  moving branch heads changed the prepared data. The iTransformer prepare now also removes the
  files it meant to and ships `itransformer.txt`.
- **time-series-library-itransformer-weather-forecast**: prepare stubs `DataEmbedding_inverted`
  in the clone's `layers/Embed.py` (705b61f). Old: the class shipped filled in there, and it is
  the answer to the four TODOs of `models/iTransformer.py`; an agent run printed it and copied it.
  New: its four lines are the same TODOs as in `models/iTransformer.py`, so the class reads as it
  does there; it is stubbed, not deleted, because `models/TimeXer.py` imports it. Only the
  checksum of `layers/Embed.py` changes (a42f4197…). Scores do not change: `models/iTransformer.py`
  defines its own class and does not import the one in `layers/Embed.py`.
- **ucr-anomaly-detection, gift-eval-short-horizon**: checksums regenerated for the 5 / 55
  datasets that prepare keeps (00279f6); they listed all 250 / 55 and never verified.
- **ucr-anomaly-detection**: the shipped `anomaly_dataset.py` reads `train/{name}.csv` (e232319),
  and its docstring no longer says the sample submission holds the labels (7ed1429; its
  checksum changes).
- **ucr-anomaly-detection labels**: the archive's indices are 1-based and inclusive, as its
  readme says ("From 1 to X is training data"; an anomaly's length is end - begin + 1)
  (e306273). The labels started one point late and missed each anomaly's first point: in 005,
  the only labelled point was the one after the distorted point. The labels now have 621, 2,
  301, 45 and 31 anomalous points (were 620, 1, 300, 44 and 30). The training split and the
  public files are unchanged; the checksum of `labels.h5` is regenerated from two identical
  rebuilds.
- **Prepare scripts**: downloads are checked (`curl -fsSL`, checked subprocesses) and failures
  raise (262a3b3); an HTTP error page was saved as the data, stomp's answers included. The
  ucr, gift-eval and MIT-BIH-WS prepare scripts work from any directory, and the PTB-XL zip has
  the name prepare expects (e232319).
- **Derived competitions**: preparing one no longer rewrites its shipped leaderboard when it is
  already the parent's (c459f01). The write failed on a read-only install.
- **optiver-realized-volatility-prediction (and its -missing and hyperparameter-search
  variants)**: the test is the 12 held-out stocks at 958 of the 3,830 time buckets (a seeded 25%),
  and `train.csv` has the 100 training stocks at the other 2,872 time buckets only (4f4a40d,
  2b5da40). The test was the 12 stocks at all 3,830 time buckets, so `train.csv` held the other
  100 stocks' targets at every test time bucket: a model using their mean per time bucket scored
  RMSPE 0.19451, better than Kaggle's winner (0.19548), and that mean times a per-stock scale
  from the book data alone scored 0.254, against 0.323 for persistence. `train.csv` now has no
  target at a test time bucket, for any stock. The book and trade files are unchanged: every
  stock at every time bucket, as on Kaggle, which gave the test period's order book but not its
  targets. The test has 11,496 rows (was 45,959) and `train.csv` 287,182 (was 382,973); scores
  on the old split are not comparable (persistence: 0.30974, was 0.32349). Only the four CSVs
  change, except that -missing also regenerates the checksums of its 200 NaN-injected parquet
  files, which could not be reproduced (writer bytes; the NaN masks are seeded and unchanged).

## Benchmark composition

- **Full split** (`experiments/splits/all.txt`): 33 challenges. The two CSDI challenges were
  dropped (be8bf8d) while their graders only raised `NotImplementedError`, and they are back now
  that they are graded (see "CSDI PM2.5" below).
- **CSDI PM2.5** (csdi-reproduce-pm25, csdi-implement-model-pm25): prepared and graded for the
  first time (9e9a345, b15b253, bb79c4e, 49aec65). Upstream had no data, no answers and a grader
  that raised `NotImplementedError`; the descriptions scored a self-reported MAE in a pickle.
  - **Data:** prepare downloads STMVL-Release (SHA-256 pinned) and lays it out as CSDI's loaders
    expect, with the evaluation targets of the test months (3, 6, 9, 12; 20,434 values) withheld
    from `pm25_ground.txt` and the normalisation statistics taken from the training months only.
  - **Score:** CSDI's own metric, the MAE in PM2.5 units over those targets (lower is better).
    The submission is `submission.csv` (`id,value`, the sample's ids in its order, finite
    values). Model weights and `result_nsample{N}.pk` are optional and not scored.
  - **Leak removed:** the `easy` branch's saved run, `generated_outputs_nsample10.pk`, held every
    test truth; prepare deletes it and clears the notebook figure that plotted a test window.
  - **Reference points** on the prepared data: per-station training mean 55.08, per-station linear
    interpolation in time 14.60, all zeros 69.66, perfect 0.
- **Registry**: lists only direct child directories (93bccb8). The feature-enhancement example's
  Hydra config was listed as a competition `conf`, and `prepare --all` failed on it.

## Validation server and agent environment

- **Uploads keep their suffix** (e232319): every upload was renamed `.csv`, so `.npy`, `.h5`,
  `.pkl` and `.py` submissions always failed validation.
- **Code submissions are only parsed**, never imported (a7ac0f3): the root server ran their
  top-level code with access to `/private`.
- **The verdict is the grader's** (bd35658): every file was reported invalid ("unexpected keyword
  argument"), None and non-finite scores count as invalid, HPS folders are checked, and single
  files of folder challenges are validated as prediction files.
- **Each upload is saved in a new private directory** (5199887): the fixed `/tmp` path let the
  agent choose what the root process read or wrote.
- **Agent image build context** (`.dockerignore`, 9766368): only the package, `environment/` and
  the top-level metadata; the whole checkout, including the raw data archive with test answers,
  was copied into the agent's image.
- **OpenHands**: `setup.py` imports correctly, so runs start (5f5ef3d); the sudo build script is
  root-owned at `/agent_build.sh` and builds once (504e1ab), as the agent could edit it.
- **python-dotenv** is installed for the agent (c674e1b): gift-eval's shipped `dataset.py` needs
  it.
- **Telemetry**: deepeval is imported only by the judge commands, after opting out; its cache
  moves to the user cache directory (e232319).

## Running agents and compiling submissions

- **Exit codes**: `run_in_container` returns the agent's exit code (it returned `run_dir`), and
  `run_agent.py` records it as `agent_exit_code`; `success` is whether it was 0 (cb149c1,
  d78c536). The exit code was never read, so a crashed `start.sh` was recorded as successful;
  then a non-zero exit made the run raise after saving its outputs.
- **`experiments/make_submission.py`** (new, e232319; 580fe02, 0596fd5, 2730ed3): one JSONL line
  per run in `metadata.json`, failed runs included, so the report counts every run; a run
  without a submission points at a missing path and is graded invalid. With several matching
  files, the most recently written one is used, as `environment/instructions.txt` says only
  the last file is evaluated. HPS lines carry the run's `code/` as `solution_file_path`;
  feature-enhancement lines carry `solution_files_path`, the submission folder if it holds
  `model.py` or else `code/`.
- **`run_agent.py`**: records `competition_id` and `run_dir` (e232319); rejects `--n-workers`
  below 1 (99d7e46), which waited forever; needs Python 3.11.4 or later (`tarfile.data_filter`),
  as on 3.11.0-3.11.3 no output was ever extracted (4ead842).
- **Extraction of run outputs**: with tarfile's "data" filter, skipping unsafe members with a
  warning (e232319, 52121d6). A plain `extractall()` recreated planted symlinks on the host, and
  then the first rejected member stopped the whole extraction. Sysbox extraction no longer loses
  outputs on large stream chunks (313405f).
- **`grade_jsonl`**: dispatches code, HPS and FE challenges to their graders (e232319, 617f82d),
  and creates the report directory with its parents before grading (77f2487).
- **Reasonableness check without an API key**: `is_reasonable` and `score` are None ("not
  evaluated"); they were True and 1.0 (b7c4072).
- **`judge --json`**: its output is no longer lost when artifacts are judged in parallel
  (f1c9eb8).
- **Code grading data** of an unsupported file type raises (a5b7817); it was UnboundLocalError
  or the previous argument's data.
- **Prepare of an uncategorised competition** raises a ValueError naming `constants.py`
  (09426ea); it was UnboundLocalError. A config that inherits only the description loads
  (4b9de66). Both are latent: no shipped competition hits them.
- **Familiarity experiment**: arXiv archives are extracted with the "data" filter, and Kaggle
  resources are scraped from the given URL (567f16b).

## Descriptions

The descriptions of 30 challenges were corrected to match their graders and prepared data (file
names, shapes, id formats, split sizes, metric definitions such as amp's SMAPE+1, scoring
tolerances), and the two missing-data variants got their own descriptions (5bc0167). The commits are
1568f6d, c81bfbb, 98b7880, fba602d, 7924a0a, 3383eff, 05da7da, 2110c7e, c011506, 3f97b77, a1ab71c,
ba20262, 7c8ab69, 1982829, 9d33b49, 2fea25d, 7184ca2, ee128f2, fe08ba1, bf7d9c8, 8eb58e3, b51204b,
21a5539, 630b7c3, 2c6a043, fa269d1, e30030e, 948a8c4, a9f20f7, f88e57a, cd734df, 617f82d, e232319,
ddba7bf, c044e1d, 9e5e756, 7c12345, b8026f5 and 9bf064a (the gift-eval metric). They change what
agents are told, not how they are graded. The three Optiver descriptions state the held-out time
buckets (848cb92).

The descriptions of LANL-Earthquake-Prediction, amp-parkinsons-disease-progression-prediction and
ashrae-energy-prediction, and a new README section, say that their medals compare scores on
TimeSeriesGym's test sets with Kaggle leaderboards scored on Kaggle's (279b011, aab3d79): LANL's
test segments are cut from Kaggle's training data, amp's test patients come from Kaggle's training
data, and ashrae's leaderboard was scored on 2017-2018, its test set here on the end of 2016.

Four more, after reviews of agent runs:

- **MIT-BIH-Arrhythmia-Weak-Supervision** (both descriptions; fffb8ed). Old: beats were "a
  256-point time series ... preprocessed and normalized", and the heuristics included an early
  R-wave and per-patient thresholds. New: each beat is a 256-sample window centred on a detected
  peak (for most beats the R peak), cut from a recording standardized as a whole, with no RR
  intervals, beat times or record ids; normal beats were kept with probability 0.15, so PVCs are a
  much larger share than in the original recordings. The early R-wave and the per-patient
  calibration are marked as not computable here, and each heuristic is to be checked on the data
  first. Reason: agents assumed the original PVC share and tried heuristics the data cannot
  support.
- **MIT-BIH-Arrhythmia** (255d75b). New: the test beats come from 8 of the 48 recordings, none of
  them in the training data. Reason: the split is by recording, and agents validated on random
  training beats.
- **context-is-key-nn5-chronos, context-is-key-moirai** (0921392). New: `cik_benchmark` and
  `run_baselines` import only with the repository root on the module search path, so scripts
  saved elsewhere need `PYTHONPATH`. Reason: agents' scripts failed with `ModuleNotFoundError`.
- **csdi-implement-model-pm25, csdi-reproduce-pm25** (9015586). New: the target mask that
  `model.evaluate` returns is empty here, all zeros, because the targets are withheld; the values
  to submit are selected by the ids of `sample_submission.csv`. Reason: an agent run selected
  them with that mask and submitted only NaN values.

## Tests

- The amp sample-score constant is 124.93847 (821b7cd).
- The slow gold-medal integration test skips challenges without a real leaderboard (696cfe0).
- `test_data.py`'s prepare test has its decorators back (119d2b3).
- `test_data.py` runs offline (00b6f4e). Its six cases called the real Kaggle API, so they failed
  on every machine without Kaggle credentials and were skipped in GitHub Actions. A fake Kaggle
  API now serves a tiny synthetic zip per competition: the download test runs the real
  `download_dataset`, and the prepare test runs the real `download_and_prepare_dataset` with each
  competition's own prepare function. Its checksums are written to a temporary file, not compared
  with the shipped ones, which belong to the real data. The integration tests still cover the
  real downloads.
- The integration tests are marked `network` (58249d6): `-m "not network"` deselects them.
- `test_code_graders.py` covers the resnet sample budget and epoch count and the stomp variant
  through the registry and `grade_code` (a296e48). 6 of its 13 tests fail before the fixes. The
  architecture check adds 5 (b402c61); 2 of its 18 tests fail without that check.
- `test_cmi_metric.py` compares the child-mind-institute metric with Kaggle's code, kept verbatim
  in the test (19a9b52, 4f3df4f; 23 tests). Its timing test fails before 19a9b52 (20,000 rows took
  52 s) and its memory test before 4f3df4f (the peak was 21 times the submission's size).

## Packaging and CI

- **CI** (6ffafd5, 125551e): a GitHub Actions workflow runs ruff 0.6.9 (`check` and
  `format --check`) and the tests not marked `network` on Python 3.11, with CPU-only PyTorch, for
  pushes to `main` and for pull requests. It needs no secrets. README.md gives the same commands
  for a local run.
- **ruff** is pinned to 0.6.9 in the dev extra (1d50468), the version CI uses. The extra allowed
  any ruff >= 0.3.0, and ruff 0.16.9 would reformat 32 files.
- **pre-commit**: `.pre-commit-config.yaml` (9bdd682) runs the `ruff` and `ruff-format` hooks of
  ruff-pre-commit v0.6.9. The README's `pre-commit install` had no configuration to install.
- **h5py** is a declared dependency, `h5py>=3.11` (510d2bd). `timeseriesgym/utils.py` imports it,
  but it was only installed because tensorflow depends on it.

## No change in behaviour

Refactors, lint and speed-ups whose outputs were checked to be unchanged: c5b82bd, 6c4c21d,
7574894, 2f55a2d, 6eae09e, ae75778, fa93243, 9a5c761, d25b02c, 675816e, cab351f, 69a6912,
8a1bdf5, 28aed04, e2979b1, 0207ff5, e3e2baf, ac3a518, 0a22d57, 78e393a, 882338c, ff0dce7,
9caa01b, f22a953, 2ac8b72, 126d7ab, a15ddc8, 362d0b8, and the test-only commits b20b786, 15abdfb
and 7ad6b1c.

- **child-mind-institute-detect-sleep-states**: the metric's matching and merging are faster, with
  bitwise the same scores (19a9b52, 4f3df4f).
  - Old: `match_detections` compared every detection with every ground truth through
    `itertuples`, about 3 ms per submitted row, and `event_detection_ap` merged the detections
    once per tolerance (10 copies). One candidate per minute on every test series, the density of
    Kaggle's winners (2,111,880 rows), took 6,209 s and 7.1 GB; 5,000,000 rows needed more than
    16 GB.
  - New: a detection is matched from the ground truths within the tolerance of it, with the same
    rule (the nearest one not yet matched, the first on ties, matched rows remembered as tuples),
    and the detections are merged and sorted once per event x series. 2,111,880 rows take 4.4 s
    at a peak RSS of 0.8 GB, and 5,000,000 rows 11.2 s at 1.9 GB.
  - Identity: `test_cmi_metric.py` compares the scores with Kaggle's code bit for bit, on random
    submissions and on edge cases: equal errors, equal ground-truth rows, rows holding a NaN,
    tied scores among competing detections, empty series, events without a step, each tolerance,
    integers past 2**52 and empty cells. Two agent submissions from the RL-environment
    evaluations were also compared. On random subsets of 10,559 to 84,475 rows, the scores are
    equal with the answers' events without a step dropped (229df50) and kept (before it). On the
    whole submissions, with them kept as Kaggle's code ran in those evaluations, both give
    0.20026379351789475 and 0.36276039952706707. A submission without any detection still raises
    `ValueError`.
