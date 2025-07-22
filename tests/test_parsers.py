import pytest
from generation_service.llm_workflows.shared.data.metadata import GenerationMetadata
from generation_service.llm_workflows.shared.tools.probability_calibration.data.beta_prior_parameters import (
    BetaDistributionParameters,
)
from generation_service.llm_workflows.shared.tools.probability_calibration.data.calibration_output import (
    CalibrationOutput,
)
from generation_service.llm_workflows.shared.types.probability import Probability
from generation_service.llm_workflows.tasks import GenerationAction
from generation_service.llm_workflows.tasks.llm_evaluation.data.output_data import (
    LlmEvaluationResult,
    LlmEvaluationOutput,
)
from generation_service.llm_workflows.tasks.named_entity_recognition.data.input_data import (
    AvailableEntities,
)
from generation_service.llm_workflows.tasks.named_entity_recognition.data.output_data import (
    NerGenerationResult,
    EntityExtractionOutput,
)

from genflow.parsers.input import parse_input_identifiers
from genflow.parsers.output import parse_result


def test_parse_input_ids():
    input_data = {
        "call_actions": {"param1": "value1", "param2": "value2"},
        "call_reason": {"param1": "value3", "param2": "value4"},
    }
    expected_output = {
        GenerationAction.CALL_ACTIONS: {"param1": "value1", "param2": "value2"},
        GenerationAction.CALL_REASON: {"param1": "value3", "param2": "value4"},
    }

    assert parse_input_identifiers(input_data) == expected_output, (
        "Parsing of task IDs did not match expected output."
    )


def test_parse_input_ids_with_invalid_id():
    with pytest.raises(ValueError) as exc_info:
        input_data = {
            "invalid_action": {"param1": "value1"},
        }
        parse_input_identifiers(input_data)


def test_parse_result():
    result = {
        GenerationAction.LLM_EVALUATION: LlmEvaluationResult(
            completion=LlmEvaluationOutput(
                evaluation=True,
                probabilist_evaluation=CalibrationOutput(
                    confidence_level=Probability(0.95),
                    lower_bound=Probability(0.782),
                    upper_bound=Probability(0.999),
                    n_positive=10,
                    posterior_parameters=BetaDistributionParameters(
                        alpha=10.5, beta=0.5
                    ),
                    prior_parameters=BetaDistributionParameters(alpha=0.5, beta=0.5),
                    sample_size=10,
                ),
                reasonings=[
                    "The provided input states 'The sky is blue.' which directly aligns with the condition that 'The sky should be blue.' Both the input and the condition describe the sky as being blue. Therefore, the condition is fully satisfied by the input as there is a direct correspondence between the two statements.",
                    "The input states 'The sky is blue.' This directly corresponds to the condition that 'The sky should be blue.' The word 'is' in the input aligns with the expectation set by 'should be' in the condition, indicating that the sky is indeed blue, as required by the condition. Therefore, the input meets the condition without any discrepancies.",
                    "The input states 'The sky is blue,' which directly describes the sky as being blue. The condition requires 'The sky should be blue.' Since the input describes the sky as blue, it fulfills the condition that the sky should be blue. Therefore, the input meets the condition as stated.",
                    "The input states 'The sky is blue.', which directly aligns with the condition 'The sky should be blue.' The input explicitly mentions that the sky is blue, which satisfies the condition that the sky should be blue. There are no discrepancies between the input and the condition, indicating a clear match.",
                    "The input states 'The sky is blue.', which directly asserts that the sky is blue. The condition requires that 'The sky should be blue.' The input explicitly confirms this condition as it asserts the sky's color as being blue. Therefore, the input meets the specified condition accurately.",
                    "The input states 'The sky is blue.', which directly corresponds to the condition 'The sky should be blue.' The input provides an observation that the sky is blue, fulfilling the requirement of the condition that the sky should be blue. Therefore, the condition is met as the input confirms the state described in the condition.",
                    "The input states 'The sky is blue.' This directly aligns with the condition that 'The sky should be blue.' Both statements assert the same fact about the color of the sky. Therefore, the input satisfies the condition as it confirms the expectation set by the condition.",
                    "The input states 'The sky is blue.' which directly matches the condition 'The sky should be blue.' The input confirms the state of the sky being blue, which aligns with the requirement specified by the condition. There is a direct correspondence between the statement in the input and the expectation in the condition, indicating that the condition is fully met.",
                    "The input states 'The sky is blue.' which directly corresponds to the condition that 'The sky should be blue.' Both the input and the condition refer to the sky being blue, with the input confirming that the sky is indeed blue. Therefore, the condition is satisfied as the input confirms the state of the sky being blue.",
                    "The input states 'The sky is blue', which directly corresponds to the condition that 'The sky should be blue.' Since the input confirms the condition by asserting that the sky is indeed blue, the condition is met. There are no discrepancies between the input and the condition.",
                ],
            ),
            metadata=GenerationMetadata(
                input_tokens=10,
                output_tokens=800,
                input_wrap_tokens=None,
                completions_count=10,
                completion_time=6.863,
                tokens_per_seconds=116,
            ),
        ),
        GenerationAction.NAMED_ENTITY_RECOGNITION: NerGenerationResult(
            completion=EntityExtractionOutput(
                entities={
                    AvailableEntities.TouchpointsEntities: [
                        {
                            "interaction": "Account Numbers",
                            "verbatim": "Client: I have two account NUMBER. One is 1234567890 and the other one, which I don't have with me right now, belongs to my wife.",
                        },
                    ],
                    AvailableEntities.Names: ["Jean Dupont", "Marie Dupont"],
                    AvailableEntities.Problems: [
                        "1234567890",
                        "9876543210",
                        "Jean Dupont",
                        "Marie Dupont",
                        "0612345678",
                        "Sports Extra package",
                    ],
                    AvailableEntities.Products: [
                        "internet connection",
                        "Wi-Fi signal",
                        "TV service",
                        "Sports Extra package",
                        "phone line",
                        "1234567890",
                        "9876543210",
                        "0612345678",
                    ],
                    AvailableEntities.PhoneNumbers: ["0612345678"],
                }
            ),
            metadata=GenerationMetadata(
                input_tokens=1036,
                output_tokens=2186,
                input_wrap_tokens=None,
                completions_count=11,
                completion_time=17.815321416975465,
                tokens_per_seconds=122,
            ),
        ),
    }

    completion, metadata = parse_result(result=result)

    assert isinstance(completion, dict), "Completion should be a dictionary."
    assert isinstance(metadata, dict), "Metadata should be a dictionary."

    assert GenerationAction.LLM_EVALUATION.value in completion.keys(), (
        "LLM_EVALUATION should be in completion."
    )
    assert GenerationAction.NAMED_ENTITY_RECOGNITION.value in completion.keys(), (
        "NAMED_ENTITY_RECOGNITION should be in completion."
    )

    assert GenerationAction.LLM_EVALUATION.value in metadata.keys(), (
        "LLM_EVALUATION should be in metadata."
    )
    assert GenerationAction.NAMED_ENTITY_RECOGNITION.value in metadata.keys(), (
        "NAMED_ENTITY_RECOGNITION should be in metadata."
    )
    assert "aggregated" in metadata.keys(), "aggregated should be in metadata."
