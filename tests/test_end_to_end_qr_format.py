import json
import os
from typing import Dict, Any

import pytest
from generation_service.llm_workflows.configuration.clients.setup_llm_client import (
    setup_llm_generation_client,
)

from genflow.main import orchestrate_generation
from tests.mock import mock_transcript


@pytest.mark.asyncio
async def test_end_to_end_queued_runner_format():
    base_url = os.getenv("LLM_GENERATION_SERVICE_BASE_URL")
    api_key = os.getenv("LLM_GENERATION_SERVICE_API_KEY")

    lm_client = setup_llm_generation_client(
        base_url=base_url, api_key=api_key, asynchronous=True, tokenizer=None
    )

    payload = {
        "b64_pub_key": "foo",
        "callback_identifier": "bar",
        "fallback_endpoint": "baz",
        "meta": {"type": "llm"},
        "base_task": "SupportTicketFromPhoneCallTranscript",
        "tasks": ["call_actions", "call_reason", "call_segments", "summary"],
        "task_parameters": {
            "custom_vocabulary": None,
            "transcript": mock_transcript,
            "company": None,
            "input_parameters": {
                "contact_reason_classification": {
                    "reasons": [
                        "Achat d'un produit",
                        "Demande d'informations",
                        "Problème technique",
                        "Demande de remboursement",
                    ],
                    "content": mock_transcript,
                },
                "rating_estimation": {"content": "Je suis très content de ce produit."},
            },
        },
        "output_language": "fr",
        "emitted_at": "0000-01-01T00:00:45+01:00",
        "delay": 1.0004281997680664,
    }

    def parse_input_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        # function that should be in the queued runner
        TRANSCRIPT_BASED_TASKS = [
            "call_actions",
            "call_reason",
            "call_segments",
            "summary",
        ]

        transcript = payload.get("task_parameters", {}).pop("transcript", None)
        custom_vocabulary = payload.get("task_parameters", {}).pop(
            "custom_vocabulary", None
        )
        company = payload.get("task_parameters", {}).pop("company", None)
        input_parameters = payload.get("task_parameters", {}).pop(
            "input_parameters", None
        )

        parameters_dict = {}

        for task in payload.get("tasks", []):
            if task in TRANSCRIPT_BASED_TASKS:
                parameters_dict[task] = {
                    "transcript": transcript,
                    "custom_vocabulary": custom_vocabulary,
                    "company": company,
                    "input_parameters": input_parameters,
                }
            else:
                parameters_dict[task] = payload.get("task_parameters", {}).get(task, {})

        return parameters_dict

    parsed_payload = parse_input_payload(payload)

    result, metadata = await orchestrate_generation(
        lm_client=lm_client, input_data=parsed_payload, timeout=300
    )

    print(" Result ".center(80, "-"))
    print(json.dumps(result, indent=4, ensure_ascii=False))
