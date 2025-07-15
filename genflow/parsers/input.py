from typing import Dict, Any

from generation_service.llm_workflows.tasks import GenerationAction


def parse_input_identifiers(
    input_data: Dict[str, Dict[str, Any]],
) -> Dict[GenerationAction, Dict[str, Any]]:
    """
    Parse the input to ensure that the keys are valid generation action identifiers.
    """
    return {
        GenerationAction.from_string(task_id): task_parameters
        for task_id, task_parameters in input_data.items()
    }
