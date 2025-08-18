# Overview

## Description
Welcome to the Timeseries Exam competition, where your statistical analysis skills will be put to the test. This competition simulates a comprehensive examination on time series concepts and methodologies.

The *Timeseries Exam* contains 746 samples of exam-style multiple choice questions, each involving single or paired time series data. This competition aims to test your understanding and ability to apply time series concepts and common statistical tools to solve problems in time series analysis.

Whether you're a student looking to test your knowledge, a professional wanting to sharpen your skills, or an enthusiast curious about time series analysis, this competition offers a challenging and educational experience for all skill levels.

To succeed in this competition, you'll need to demonstrate your proficiency in analyzing temporal data patterns, identifying appropriate statistical tests, and making informed decisions based on time series evidence.

## Evaluation

### Metric

Submissions are evaluated based on their accuracy, the percentage of questions answered correctly. Each question has equal weight in the final score.

### Submission Format

The submission format for the competition is a csv file with the following format:

```
question_id,answer
0,0
1,0
2,0
3,0
etc.
```

# Dataset Description

In this competition your task is to solve multiple-choice questions based on time series analysis. Each question involves either a single time series or a pair of time series, testing your understanding of time series concepts and statistical methodologies.

> **⚠️ IMPORTANT NOTE ⚠️**: It is extremely important to realize that this competition has no train and test split. Each sample in the provided question csv file represents a question. You should try to solve each question independently as in an exam. Good luck!

## File and Data Field Descriptions

- **qa_dataset.csv** - This file contains the exam with 746 multiple-choice questions.
    - `question_id` - id of the sample
    - `ts1` - A list that contains the primary time series data for analysis.
    - `ts2` - A list that contains a second time series (if applicable) or NaN (if the question involves only one time series).
    - `question` - The exam question to be answered.
    - `options` - List of multiple-choice options to select from.
    - `question_hint` - A textual hint providing additional context for the question.
    - `relevant_concept` - List of relevant time series concepts applicable to the question.

- **sample_submission.csv** - A submission file in the correct format.
    - `question_id` - Unique identifier for each question in the test set.
    - `answer` - Your predicted answer for each question, zero-indexed location of correct answer in the provided option list.
