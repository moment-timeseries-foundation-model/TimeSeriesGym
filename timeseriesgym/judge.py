import builtins
import concurrent.futures
import json
import logging
import os
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from io import StringIO

import yaml
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv

from timeseriesgym.utils import get_logger

logger = get_logger(__name__)
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

logging.getLogger("openai._client").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("deepeval").setLevel(logging.WARNING)


def llm_judge(config_path, model_name, json_output=False):
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    artifacts = config.get("artifacts", [])
    global_criteria = config.get("criteria", [])

    if not artifacts:
        logger.error("Config file must contain 'artifacts' section")
        return

    original_print = builtins.print

    def silent_print(*args, **kwargs):
        if (
            args
            and isinstance(args[0], str)
            and not args[0].startswith("✨")
            and "False !!!!!!!!!!!!" not in args[0]
        ):
            original_print(*args, **kwargs)

    def evaluate_artifact(artifact):
        path = artifact.get("path")
        artifact_type = artifact.get("type")

        artifact_criteria = artifact.get("criteria", global_criteria)

        if not path or not artifact_criteria:
            logger.warning(f"Skipping artifact {path}: missing path or criteria")
            return None

        try:
            with open(path) as f:
                content = f.read()

            test_case = LLMTestCase(
                input=f"Evaluate {artifact_type} artifact", actual_output=content
            )

            criteria_list = []
            for criterion in artifact_criteria:
                name = criterion.get("name")
                description = criterion.get("description")
                threshold = criterion.get("threshold", 0.6)

                criterion_eval = GEval(
                    name=name,
                    criteria=description,
                    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
                    threshold=threshold,
                    model=model_name,
                    verbose_mode=False,
                )
                criteria_list.append(criterion_eval)

            if criteria_list:
                if not json_output:
                    logger.info(f"Evaluating artifact: {path}")
                results = []

                builtins.print = silent_print

                try:
                    for criterion in criteria_list:
                        stdout_capture = StringIO()
                        stderr_capture = StringIO()
                        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                            criterion.measure(test_case)

                        results.append(
                            {
                                "name": criterion.name,
                                "score": criterion.score,
                                "reason": criterion.reason,
                                "passed": criterion.score >= criterion.threshold,
                            }
                        )
                finally:
                    builtins.print = original_print

                return {"path": path, "type": artifact_type, "results": results}

        except Exception as e:
            if not json_output:
                logger.error(f"Error evaluating artifact {path}: {e}")
            return None

    if not json_output:
        logger.info("Starting evaluations with deepeval in parallel...")

    builtins.print = silent_print

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_artifact = {
                executor.submit(evaluate_artifact, artifact): artifact for artifact in artifacts
            }
            evaluation_results = []

            for future in concurrent.futures.as_completed(future_to_artifact):
                artifact = future_to_artifact[future]
                try:
                    result = future.result()
                    if result:
                        evaluation_results.append(result)
                except Exception as e:
                    if not json_output:
                        logger.error(f"Error evaluating {artifact.get('path')}: {e}")
    finally:
        builtins.print = original_print

    if json_output:
        final_results = {
            "summary": {
                "total_artifacts": len(evaluation_results),
                "total_criteria": sum(len(result["results"]) for result in evaluation_results),
                "passed_criteria": sum(
                    sum(1 for m in result["results"] if m["passed"])
                    for result in evaluation_results
                ),
            },
            "artifacts": evaluation_results,
        }
        print(json.dumps(final_results, indent=2))
        return evaluation_results

    logger.info(f"Evaluation complete. Processed {len(evaluation_results)} artifacts.")

    if evaluation_results:
        logger.info("Detailed evaluation results:")
        for result in evaluation_results:
            logger.info(f"\n{'='*50}")
            logger.info(f"Artifact: {result['path']}")
            logger.info(f"Type: {result['type']}")
            logger.info(f"{'='*50}")

            for criterion_result in result["results"]:
                logger.info(f"\nCriterion: {criterion_result['name']}")
                logger.info(
                    f"Score: {criterion_result['score']:.2f} (Passed: {criterion_result['passed']})"
                )
                logger.info(f"Reason: {criterion_result['reason']}")
                logger.info(f"{'-'*50}")

        total_criteria = sum(len(result["results"]) for result in evaluation_results)
        passed_criteria = sum(
            sum(1 for m in result["results"] if m["passed"]) for result in evaluation_results
        )

        logger.info("\nSummary Statistics:")
        logger.info(f"Total artifacts processed: {len(evaluation_results)}")
        logger.info(f"Total criteria evaluated: {total_criteria}")
        logger.info(
            f"Passed criteria: {passed_criteria} ({(passed_criteria/total_criteria)*100:.1f}%)"
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"evaluation_results_{timestamp}.json"
        with open(result_file, "w") as f:
            json.dump(evaluation_results, f, indent=2)
        logger.info(f"Detailed results exported to {result_file}")

    return evaluation_results
