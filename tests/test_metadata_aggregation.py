import pytest
from generation_service.llm_workflows.shared.data.metadata import GenerationMetadata

from genflow.parsers.output import aggregate_metadata


def test_aggregate_metadata():
    metadata = [
        GenerationMetadata(
            input_tokens=100,
            output_tokens=50,
            input_wrap_tokens=10,
            completions_count=1,
            completion_time=2.0,
            out_in_ratio=50 / 100,
            tokens_per_seconds=25,
        ),
        GenerationMetadata(
            input_tokens=200,
            output_tokens=100,
            input_wrap_tokens=20,
            completions_count=2,
            completion_time=3.0,
            out_in_ratio=100 / 200,
            tokens_per_seconds=33,
        ),
        GenerationMetadata(
            input_tokens=150,
            output_tokens=75,
            input_wrap_tokens=15,
            completions_count=1,
            completion_time=1.0,
            out_in_ratio=75 / 150,
            tokens_per_seconds=75,
        ),
    ]

    aggregated = aggregate_metadata(metadata)

    assert aggregated.get("input_tokens") == 450, "Input tokens aggregation failed."
    assert aggregated.get("output_tokens") == 225, "Output tokens aggregation failed."
    assert aggregated.get("input_wrap_tokens") == 45, (
        "Input wrap tokens aggregation failed."
    )
    assert aggregated.get("completions_count") == 4, (
        "Completions count aggregation failed."
    )
    assert aggregated.get("completion_time") == 3.0, (
        "Completion time aggregation failed."
    )
    assert aggregated.get("out_in_ratio") == 225 / 450, (
        "Out/In ratio aggregation failed."
    )
    assert 74.9 < aggregated.get("tokens_per_seconds") < 75.1, (
        "Tokens per second aggregation failed."
    )


def test_aggregate_metadata_invalid_type():
    with pytest.raises(TypeError):
        aggregate_metadata([{"input_tokens": 100}])
