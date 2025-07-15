from unittest.mock import patch

from generation_service.llm_workflows.tasks import GenerationAction

from genflow.tools.runnable_analysis import (
    is_action_runnable,
    get_runnable_actions,
)


def test_is_action_runnable_when_no_deps():
    deps_status = ({}, None)  # test with empty set and None
    for dependencies in deps_status:
        result = {}
        assert is_action_runnable(
            action_id=GenerationAction.CALL_ACTIONS,
            dependencies=dependencies,
            result=result,
        ), "Action should be runnable when no dependencies are defined."


def test_is_action_runnable_when_deps_ended():
    dependencies = {GenerationAction.CALL_ACTIONS, GenerationAction.CALL_REASON}
    result = {
        GenerationAction.CALL_ACTIONS: {"foo": "bar"},
        GenerationAction.CALL_REASON: None,
    }
    assert is_action_runnable(
        action_id=GenerationAction.CRITERION_GENERATION,
        dependencies=dependencies,
        result=result,
    ), "Action should be runnable when dependencies are completed."


def test_is_action_runnable_false_when_in_results():
    dependencies = None
    result = {
        GenerationAction.CALL_ACTIONS: {"foo": "bar"},
        GenerationAction.CALL_REASON: None,
    }
    assert not is_action_runnable(
        action_id=GenerationAction.CALL_ACTIONS,
        dependencies=dependencies,
        result=result,
    ), "Action should not be runnable twice."


def test_is_action_runnable_false_due_to_deps():
    dependencies = {GenerationAction.CALL_ACTIONS, GenerationAction.CALL_REASON}
    result = {
        GenerationAction.CALL_ACTIONS: {"foo": "bar"},
    }
    assert not is_action_runnable(
        action_id=GenerationAction.LLM_EVALUATION,
        dependencies=dependencies,
        result=result,
    ), "Action should not be runnable when not all dependencies are completed."


def test_get_runnable_actions():
    actions = {
        GenerationAction.CALL_ACTIONS: {"param1": "value1"},
        GenerationAction.CALL_REASON: {"param2": "value2"},
        GenerationAction.CALL_SEGMENTS: {"param2": "value2"},
        GenerationAction.CUSTOM_GENERATION: None,
    }
    result = {
        GenerationAction.CALL_ACTIONS: {"foo": "bar"},
    }
    runnable_actions = {
        GenerationAction.CALL_SEGMENTS: {"param2": "value2"},
        GenerationAction.CUSTOM_GENERATION: None,
    }

    def mock_dependencies(action: GenerationAction) -> set[GenerationAction] | None:
        if action == GenerationAction.CALL_SEGMENTS:
            return {GenerationAction.CALL_ACTIONS}
        elif action == GenerationAction.CALL_REASON:
            return {GenerationAction.CUSTOM_GENERATION}
        return None

    with patch(
        target="genflow.tools.runnable_analysis.get_dependencies",
        side_effect=mock_dependencies,
    ):
        assert (
            get_runnable_actions(actions=actions, result=result) == runnable_actions
        ), "Runnable actions did not match expected output."
