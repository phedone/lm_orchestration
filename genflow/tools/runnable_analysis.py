from typing import Any, Dict, Set

from generation_service.llm_workflows.shared.data.dependencies_getter import (
    get_dependencies,
)
from generation_service.llm_workflows.tasks import GenerationAction


def is_action_runnable(
    action_id: GenerationAction,
    dependencies: Set[GenerationAction] | None,
    result: dict[GenerationAction, Any],
) -> bool:
    """
    Action is runnable if:
    - it is not already in the result
    - all dependencies are met (if any)
    """

    if action_id in result.keys():
        return False

    elif dependencies:
        return all(dep in result.keys() for dep in dependencies)
    return True


def get_runnable_actions(
    actions: Dict[GenerationAction, Dict[str, Any]],
    result: Dict[GenerationAction, Dict[str, Any]],
) -> Dict[GenerationAction, Dict[str, Any]]:
    """
    Returns a dictionary of actions_id: parameters that are runnable based on their dependencies and the current result.

    Args:
        actions (Dict[GenerationAction, Dict[str, Any]]): A dictionary of actions with their parameters.
        result (Dict[GenerationAction, Dict[str, Any]]): The current result dictionary containing completed actions.

    Returns: Dict[GenerationAction, Dict[str, Any]]: A dictionary of runnable actions with their parameters.
    """
    return {
        action_id: action_parameters
        for action_id, action_parameters in actions.items()
        if is_action_runnable(
            dependencies=get_dependencies(action=action_id),
            result=result,
            action_id=action_id,
        )
    }
