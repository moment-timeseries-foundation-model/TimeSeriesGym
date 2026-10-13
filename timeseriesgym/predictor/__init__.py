"""The predictor protocol of gift-eval-short-horizon and time-series-library-long-horizon-forecast.

In these challenges the agent submits a predictor (a directory with predict.py), not
predictions: the test labels are not in the public data, and the grader runs the predictor on
the hidden test windows, in time order.

- protocol.py: windows, the round schedule and the wire format (stdlib and numpy).
- runner.py: the predictor process: imports predict.py and answers the grader's calls.
- launch.py: how a predictor process is started (the interface a sandbox implements).
- windows.py: the windows of each challenge, built with the challenge's own data code.
- contract.py: the format of a predictor directory.
- grading.py: the grader (test windows from the private data, run, score).
- check.py: the agent's local check, on validation windows of the public data.
- prepare.py: helpers of the challenges' prepare.py; leakscan.py: the value-overlap scan.

protocol, runner, windows, contract, launch and check need neither timeseriesgym nor the
private data: the prepare copies them into the public data (`predictor_tools/`), which is why
they import each other relatively.
"""
