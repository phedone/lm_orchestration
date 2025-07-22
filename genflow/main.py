import asyncio
import logging
import time
from asyncio import TaskGroup
from typing import Any, Dict, Optional

from generation_service.llm_workflows.configuration.clients.baml_client import (
    BamlGenerationClient,
)
from generation_service.llm_workflows.shared.data.available_languages import (
    AvailableLanguage,
)
from generation_service.llm_workflows.shared.interfaces.generation_result import (
    GenerationResult,
)
from generation_service.llm_workflows.tasks import GenerationAction

from genflow.parsers.input import parse_input_identifiers
from genflow.parsers.output import ResultMetadataTuple, parse_result
from genflow.tools.run_generation_action import run_generation_action
from genflow.tools.runnable_analysis import get_runnable_actions


async def run_actions(
    lm_client: BamlGenerationClient,
    runnable_actions: Dict[GenerationAction, Dict[str, Any]],
    language: AvailableLanguage,
) -> Dict[GenerationAction, GenerationResult]:
    async with TaskGroup() as group:
        actions = {
            action_id: group.create_task(
                run_generation_action(
                    lm_client=lm_client,
                    action_id=action_id,
                    action_parameters=action_parameters,
                    language=language,
                )
            )
            for action_id, action_parameters in runnable_actions.items()
        }

    return {action_id: task.result() for action_id, task in actions.items()}


async def orchestrate_generation(
    lm_client: BamlGenerationClient,
    input_data: Dict[str, Dict[str, Any]],
    timeout: float = 300,
    logger: Optional[logging.Logger] = None,
) -> ResultMetadataTuple:
    """
    Orchestrate the generation process by parsing the input data and executing tasks respecting their dependencies.

    Args:
        lm_client (BamlGenerationClient): The client to use for executing the generation tasks.
        input_data (Dict[str, Dict[str, Any]]): A dictionary where keys are task identifiers and values are task parameters.
        timeout (float): Maximum time (seconds) to wait for the generation tasks to complete.
        logger (Optional[logging.Logger]): Logger to use.

    Returns:
        ResultMetadataTuple: A tuple containing aggregated metadata and results from the generation tasks.
    """

    is_logger_enabled = logger is not None

    language = AvailableLanguage.from_string(s=input_data.pop("language", "en"))

    if not input_data:
        raise ValueError("Input data cannot be empty.")

    parsed_input = parse_input_identifiers(input_data=input_data)

    result = {}
    all_tasks_completed = False

    if is_logger_enabled:
        logger.info(
            f"Starting orchestration with {len(parsed_input)} actions and timeout set to {timeout:.2f} seconds.",
        )

    elapsed_time = 0.0
    start_time = time.time()

    while not all_tasks_completed:
        elapsed_time = time.time() - start_time
        if elapsed_time >= timeout:
            raise asyncio.TimeoutError(
                f"Timeout reached after {elapsed_time:.2f} seconds. Not all tasks completed."
            )

        runnable_actions = get_runnable_actions(
            actions=parsed_input,
            result=result,
        )

        if not runnable_actions:
            raise ValueError(
                "No runnable actions found. This may indicate circular dependencies between tasks."
            )

        action_results = await run_actions(
            lm_client=lm_client,
            runnable_actions=runnable_actions,
            language=language,
        )

        if is_logger_enabled:
            logger.info(
                f"Executed {len(action_results)} actions in {time.time() - start_time:.2f} seconds. {len(input_data) - len(result)} actions remaining.",
            )

        result.update(action_results)

        all_tasks_completed = result.keys() == parsed_input.keys()

    if is_logger_enabled:
        logger.info(
            f"All tasks completed in {elapsed_time:.2f} seconds, within the timeout of {timeout:.2f} seconds."
        )

    output, metadata = parse_result(result=result)

    return output, metadata
