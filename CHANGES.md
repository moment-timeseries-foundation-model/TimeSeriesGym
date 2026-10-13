# Changes affecting scores, validity and behaviour

This file lists every change on this branch, relative to upstream `main` (520fea8), that changes
a score, whether a submission is valid, the data a challenge is prepared with, or how the tools
behave. Each entry gives the old and new behaviour, the reason, and the commit. Refactors, lint
and speed-ups that change no output are listed at the end.

Results produced before these changes are not directly comparable for the challenges named
under "Scores and validity", "Hyperparameter search", "Feature enhancement" and "Prepared data".

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

## Medals and leaderboards

- **Placeholder leaderboards**: one-row "dummy" leaderboards are ignored, so these 10 challenges
  award no medals: context-is-key-moirai, context-is-key-nn5-chronos, gift-eval-short-horizon,
  moment-anomaly-detection-score, ptb-xl-classification-challenge and its HPS and FE variants,
  time-series-exam-1-1, time-series-library-itransformer-weather-forecast, ucr-anomaly-detection
  (229df50). With one row, every score >= 0 was gold, even for lower-is-better metrics, and
  run-group medal totals counted every valid submission on them as gold.
- **Non-finite scores** never earn a medal (229df50).
- **`awards_medals`** is now loaded into `Competition` but stays informational, as on Kaggle:
  new-york-city-taxi-fare-prediction and spaceship-titanic (`false`, real leaderboards) keep
  their medals (696cfe0).

## Prepared data and checksums

Rebuilding these challenges gives different files; results on the old data are not comparable.

- **amp-parkinsons-disease-progression-prediction**: one answer per `prediction_id`, joined on
  visit and UPDRS part (21a2a7e). Each id was paired with every part rated at the visit (11,004
  rows for 3,183 ids), so a perfect forecast scored SMAPE+1 72.97; it now scores 0. The sample
  submission scores 124.93847 (was 129.7763). Only `ground_truth.csv` and
  `sample_submission.csv` change.
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
- **gift-eval-short-horizon**: unchanged. Its documented source is the Git LFS bundle, whose zip
  matches `checksums.yaml`; the Hugging Face download path never matched it.
- **m5-forecasting-accuracy**: the public `sell_prices.csv` has every training week (6,597,201
  rows; was 4,088,360) (d64a36f). Days were compared as strings, so d_2..d_9 and d_200..d_999
  counted as later than d_1913. Scores do not change: the grader reads prices from the answers.
- **LANL-Earthquake-Prediction**: test segment ids come from a seeded generator, duplicates are
  rejected, and the checksums are regenerated (7a8862f). Every prepare named the 2,500 segments
  differently, so checksums could never verify.
- **Cloned task repositories**: shipped without `.git` (1568f6d) and pinned to commits (8dbe2b8;
  MOMENT to 37a8bde in 00279f6). The history held the reference solution (iTransformer), and
  moving branch heads changed the prepared data. The iTransformer prepare now also removes the
  files it meant to and ships `itransformer.txt`.
- **ucr-anomaly-detection, gift-eval-short-horizon**: checksums regenerated for the 5 / 55
  datasets that prepare keeps (00279f6); they listed all 250 / 55 and never verified.
- **ucr-anomaly-detection**: the shipped `anomaly_dataset.py` reads `train/{name}.csv` (e232319),
  and its docstring no longer says the sample submission holds the labels (7ed1429; its
  checksum changes).
- **Prepare scripts**: downloads are checked (`curl -fsSL`, checked subprocesses) and failures
  raise (262a3b3); an HTTP error page was saved as the data, stomp's answers included. The
  ucr, gift-eval and MIT-BIH-WS prepare scripts work from any directory, and the PTB-XL zip has
  the name prepare expects (e232319).
- **Derived competitions**: preparing one no longer rewrites its shipped leaderboard when it is
  already the parent's (c459f01). The write failed on a read-only install.

## Benchmark composition

- **Full split** (`experiments/splits/all.txt`): 31 challenges; the two CSDI challenges are
  dropped (be8bf8d). Their graders only raise `NotImplementedError` and their prepare writes no
  answers. They stay registered.
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

## Agents' network and API keys

Agent runs made before these changes had the internet and their provider keys; runs made after
have neither, so their results are not directly comparable, least of all on the challenges that
install packages or download weights at run time (agents/README.md lists them).

- **No network** (3299c6d): agent containers were on Docker's default network, so agent code could
  download the public data the challenges are built from (Kaggle, UCR, PhysioNet, Hugging Face,
  this repository) and send anything anywhere. They now get Docker's `none` network: loopback
  only. `run_agent.py --agent-network bridge` (or the name of an operator's network) gives a run
  group a network; `metadata.json` records it as `agent_network`. `host` and `container:<id>` are
  refused, and a container config that sets the network is an error.
- **API keys stay on the host** (0348a4c, 3c96469, b92f1a1, 9d3fb13): `agents/run.py` put the
  agent config's provider keys into the container's environment, where agent code could read
  them. Each run now starts a model proxy on the host (`agents/model_proxy.py`) that holds the
  keys and forwards only POST requests to the providers' model endpoints (OpenAI, Anthropic,
  OpenRouter, Gemini's OpenAI-compatible API) for the models in the agent's config, up to a
  request and a time budget (AIDE 1000 requests and 16200 s, MLAgentBench 10000 and 5400 s,
  OpenHands 2000 and 23400 s). The container gets base URLs with a per-run token and dummy keys,
  and reaches the proxy through a relay on `127.0.0.1:18080`. Configs give keys under
  `model_proxy`; `${{ secrets.* }}` in `env_vars` or `kwargs` is refused. The proxy's log
  (`<run>/model_proxy.jsonl`) holds counts, status codes and sizes, no content or keys.
- **AIDE's Claude, Llama and Gemini variants** get an OpenAI key too (b92f1a1): their feedback
  model is `gpt-4o-2024-08-06`, but their configs passed only the other provider's key. The
  Llama (OpenRouter) and Gemini variants cannot reach their models until the AIDE fork reads
  `OPENROUTER_BASE_URL` and uses Gemini's OpenAI-compatible API at `GEMINI_BASE_URL`.
- **OpenHands builds when its image is built** (d98af77): `make build`, `playwright install` and
  the NVIDIA container toolkit ran at agent start, the first two through a sudo rule for
  `/agent_build.sh` (504e1ab), which is gone with the script. litellm uses its bundled model price
  list. OpenHands' Docker runtime still builds its sandbox image at agent start, so OpenHands
  needs `--agent-network bridge` for now.
- **MLAgentBench's tiktoken encodings** are fetched when its image is built (d98af77); tiktoken
  downloaded them on first use.
- **Tests**: `tests/integration/test_agent_network.py` checks the isolation with a throwaway
  container and needs Docker (3299c6d).

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
ddba7bf, c044e1d, 9e5e756, 7c12345 and b8026f5. They change what agents are told, not how they are
graded.

## Tests

- The amp sample-score constant is 124.93847 (821b7cd).
- The slow gold-medal integration test skips challenges without a real leaderboard (696cfe0).
- `test_data.py`'s prepare test has its decorators back (119d2b3); like the download test, it
  needs Kaggle credentials.

## No change in behaviour

Refactors, lint and speed-ups whose outputs were checked to be unchanged: c5b82bd, 6c4c21d,
7574894, 2f55a2d, 6eae09e, ae75778, fa93243, 9a5c761, d25b02c, 675816e, cab351f, 69a6912,
8a1bdf5, 28aed04, e2979b1, 0207ff5, e3e2baf, ac3a518, 0a22d57, 78e393a, 882338c, ff0dce7,
9caa01b, f22a953, 2ac8b72, 126d7ab, a15ddc8, 362d0b8, and the test-only commits b20b786, 15abdfb
and 7ad6b1c.
