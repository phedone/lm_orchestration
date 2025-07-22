from typing import Dict, Any, Tuple, Iterable, cast

import numpy as np
from generation_service.llm_workflows.shared.data.metadata import GenerationMetadata
from generation_service.llm_workflows.shared.interfaces.generation_result import (
    GenerationResult,
)
from generation_service.llm_workflows.tasks import GenerationAction

ResultMetadataTuple = Tuple[Dict[str, Any], Dict[str, Any]]


def aggregate_metadata(
    all_metadata: Iterable[GenerationMetadata],
) -> Dict[str, int | float]:
    """
    Aggregate metadata from multiple GenerationMetadata instances.

    Args:
        all_metadata (Iterable[GenerationMetadata]): An iterable of GenerationMetadata instances to aggregate.

    Returns: The aggregated metadata as a dictionary.
    """

    if not all(isinstance(meta, GenerationMetadata) for meta in all_metadata):
        raise TypeError(
            "All GenerationMetadata instances must be of type GenerationMetadata"
        )

    aggregated_metadata = {
        "input_tokens": None,
        "output_tokens": 0,
        "input_wrap_tokens": None,
        "completions_count": 0,
        "completion_time": 0.0,
    }

    for metadata in all_metadata:
        if metadata.input_tokens:
            aggregated_metadata["input_tokens"] = metadata.input_tokens + (
                aggregated_metadata["input_tokens"] or 0
            )
        aggregated_metadata["output_tokens"] += metadata.output_tokens
        if metadata.input_wrap_tokens:
            aggregated_metadata["input_wrap_tokens"] = metadata.input_wrap_tokens + (
                aggregated_metadata["input_wrap_tokens"] or 0
            )

        aggregated_metadata["completions_count"] += metadata.completions_count
        aggregated_metadata["completion_time"] = np.max(
            max(metadata.completion_time, aggregated_metadata["completion_time"])
        )  # using max due to asynchronicity of tasks

    aggregated_metadata["out_in_ratio"] = (
        (aggregated_metadata["output_tokens"] / aggregated_metadata["input_tokens"])
        if aggregated_metadata["input_tokens"]
        else None
    )

    aggregated_metadata["tokens_per_seconds"] = aggregated_metadata["output_tokens"] / (
        aggregated_metadata["completion_time"] + 1e-6
    )

    return aggregated_metadata


def parse_result(
    result: Dict[GenerationAction, GenerationResult],
) -> ResultMetadataTuple:
    """
    Parse the output data to extract metadata and results and return them in the queued runner format.

    Args:
        result: A dictionary where keys are GenerationAction identifiers and values are GenerationResult instances.

    Returns:
        A tuple containing two dictionaries:
        - The first dictionary contains the results for each action.
        - The second dictionary contains metadata for each action.
    """
    metadata = {}
    completion = {}
    all_metadata = [None] * len(result)

    for result_idx, (action_id, action_result) in enumerate(result.items()):
        completion[action_id.value] = action_result.completion.to_dict()
        metadata[action_id.value] = action_result.metadata.to_dict()
        all_metadata[result_idx] = action_result.metadata

    metadata["aggregated"] = aggregate_metadata(
        all_metadata=cast(Iterable[GenerationMetadata], all_metadata)
    )

    return completion, metadata
