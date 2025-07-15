import asyncio
import copy
import os
from typing import Dict, Any
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from generation_service.baml.baml_client.types import (
    ActionActor,
    PhoneCallAction,
)
from generation_service.llm_workflows.configuration.clients.baml_client import (
    BamlGenerationClient,
)
from generation_service.llm_workflows.configuration.clients.setup_llm_client import (
    setup_llm_generation_client,
)
from generation_service.llm_workflows.shared.data.available_languages import (
    AvailableLanguage,
)
from generation_service.llm_workflows.shared.data.metadata import GenerationMetadata
from generation_service.llm_workflows.shared.interfaces.generation_result import (
    GenerationResult,
)
from generation_service.llm_workflows.tasks import GenerationAction
from generation_service.llm_workflows.tasks.call_actions.data.call_actions_output import (
    CallActionsResult,
    CallActionsOutput,
)

from genflow.main import orchestrate_generation
from tests import test_logger
from tests.mock import mock_transcript


class TestOrchestration:
    def setup_method(self) -> None:
        self.mock_lm_client = setup_llm_generation_client(
            base_url="mock_url", api_key="mock_key", asynchronous=True, tokenizer=None
        )

    @staticmethod
    async def mock_run_generation_action(
        lm_client: BamlGenerationClient,
        action_id: GenerationAction,
        action_parameters: Dict[str, Any],
        language: AvailableLanguage,
    ) -> GenerationResult:
        result = CallActionsResult(
            completion=CallActionsOutput(
                actions_extraction_exhaustive_reasoning_detailed_paragraphs="During the call, the client reported an issue with their internet not functioning. The agent needs to troubleshoot this problem by guiding the client through a series of diagnostic and corrective steps. The conversation likely involved the agent instructing the client to check the modem, ensuring it is powered on and connected properly. If the issue persists, the agent may have suggested restarting the modem or checking the cables. The agent might have also verified the customer's account status to ensure there are no service outages or billing issues affecting the service. After the call, the agent is responsible for documenting the issue and any steps taken during the troubleshooting process. If the problem remains unresolved, the agent may schedule a technician visit or escalate the issue to a higher support tier. The client, on the other hand, is expected to perform the troubleshooting steps suggested by the agent during the call and report back the outcomes.",
                actions=[
                    PhoneCallAction(
                        title="Verify Client's Account and Service Status",
                        why="To ensure there are no outages or account-related issues affecting the service.",
                        who=ActionActor.Agent,
                        steps=[
                            "Check the client's account for any service outages or billing issues."
                        ],
                        completed=False,
                    )
                ],
            ),
            metadata=GenerationMetadata(
                input_tokens=15,
                output_tokens=622,
                input_wrap_tokens=438,
                completions_count=1,
                completion_time=4.439,
                tokens_per_seconds=140,
            ),
        )

        call_actions_result = copy.deepcopy(result)
        call_reason_result = copy.deepcopy(result)
        call_segments_result = copy.deepcopy(result)

        action_results = {
            GenerationAction.CALL_ACTIONS: call_actions_result,
            GenerationAction.CALL_REASON: call_reason_result,
            GenerationAction.CALL_SEGMENTS: call_segments_result,
        }

        return action_results[action_id]

    @pytest.mark.asyncio
    async def test_successful_orchestration(self):
        params = {
            "transcript": "This is a test transcript.",
            "company": "TestCorp",
            "custom_vocabulary": None,
        }

        input_data = {
            "call_actions": params,
            "call_reason": params,
            "call_segments": params,
            "language": "fr",
        }

        mock_dependencies = {
            GenerationAction.CALL_ACTIONS: None,
            GenerationAction.CALL_REASON: None,
            GenerationAction.CALL_SEGMENTS: {GenerationAction.CALL_ACTIONS},
        }

        with patch(
            target="genflow.tools.runnable_analysis.get_dependencies",
            side_effect=lambda action: mock_dependencies[action],
        ):
            with patch(
                target="genflow.main.run_generation_action",
                side_effect=self.mock_run_generation_action,
            ):
                result, metadata = await orchestrate_generation(
                    lm_client=self.mock_lm_client,
                    input_data=input_data,
                    timeout=300,
                    logger=test_logger,
                )

        for key in input_data.keys():
            if key != "language":
                assert key in result.keys(), (
                    f"{key} was expected to be in the result dict"
                )
                assert key in metadata.keys(), (
                    f"{key} was expected to be in the metadata dict"
                )
        assert "aggregated" in metadata.keys(), "'Aggregated' should be in metadata"

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        with pytest.raises(asyncio.TimeoutError):
            await orchestrate_generation(
                lm_client=self.mock_lm_client,
                input_data={
                    "call_actions": {"param1": "value1"},
                    "language": "en",
                },
                timeout=-1,
                logger=test_logger,
            )

    @pytest.mark.asyncio
    async def test_empty_input_data(self):
        with pytest.raises(ValueError):
            await orchestrate_generation(
                lm_client=self.mock_lm_client,
                input_data={},
                timeout=300,
                logger=test_logger,
            )

    @pytest.mark.asyncio
    async def test_circular_dependencies(self):
        mock_dependencies = {
            GenerationAction.CALL_ACTIONS: {GenerationAction.CALL_REASON},
            GenerationAction.CALL_REASON: {GenerationAction.CALL_ACTIONS},
            GenerationAction.CALL_SEGMENTS: None,
        }

        params = {
            "transcript": "This is a test transcript.",
            "company": "TestCorp",
            "custom_vocabulary": None,
        }

        input_data = {
            "call_actions": params,
            "call_reason": params,
            "call_segments": params,
            "language": "fr",
        }

        with pytest.raises(ValueError):
            with patch(
                target="genflow.tools.runnable_analysis.get_dependencies",
                side_effect=lambda action: mock_dependencies[action],
            ):
                with patch(
                    target="genflow.main.run_generation_action",
                    side_effect=self.mock_run_generation_action,
                ):
                    result, metadata = await orchestrate_generation(
                        lm_client=self.mock_lm_client,
                        input_data=input_data,
                        timeout=300,
                        logger=test_logger,
                    )

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_execute_tasks(self):
        load_dotenv()
        base_url = os.getenv("LLM_GENERATION_SERVICE_BASE_URL")
        api_key = os.getenv("LLM_GENERATION_SERVICE_API_KEY")

        lm_client = setup_llm_generation_client(
            base_url=base_url, api_key=api_key, asynchronous=True, tokenizer=None
        )

        params = {
            "transcript": mock_transcript,
            "company": "XYZ Internet Services",
            "custom_vocabulary": None,
        }

        input_data = {
            "call_actions": params,
            "call_reason": params,
            "call_segments": params,
            "language": "fr",
        }
        mock_dependencies = {
            GenerationAction.CALL_ACTIONS: None,
            GenerationAction.CALL_REASON: None,
            GenerationAction.CALL_SEGMENTS: {GenerationAction.CALL_ACTIONS},
        }

        with patch(
            target="genflow.tools.runnable_analysis.get_dependencies",
            side_effect=lambda action: mock_dependencies[action],
        ):
            result, metadata = await orchestrate_generation(
                lm_client=lm_client,
                input_data=input_data,
                timeout=300,
                logger=test_logger,
            )

        for key in input_data.keys():
            if key != "language":
                assert key in result.keys(), (
                    f"{key} was expected to be in the result dict"
                )
                assert key in metadata.keys(), (
                    f"{key} was expected to be in the metadata dict"
                )
        assert "aggregated" in metadata.keys(), "'Aggregated' should be in metadata"
