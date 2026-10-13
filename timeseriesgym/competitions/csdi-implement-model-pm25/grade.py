"""Graded as csdi-reproduce-pm25: the same evaluation targets, submission format and metric
(mean absolute error, lower is better)."""

from timeseriesgym.utils import import_fn

grade = import_fn("timeseriesgym.competitions.csdi-reproduce-pm25.grade:grade")
