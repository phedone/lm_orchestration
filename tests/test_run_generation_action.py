from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from generation_service.llm_workflows.configuration.clients.setup_llm_client import (
    setup_llm_generation_client,
)
from generation_service.llm_workflows.shared.data.available_languages import (
    AvailableLanguage,
)
from generation_service.llm_workflows.tasks import GenerationAction

from genflow.tools.run_generation_action import run_generation_action

from generation_service.llm_workflows.shared.data.transcript_based_input_parameters import (
    TranscriptBasedInputParameters,
)
from generation_service.llm_workflows.tasks.RAG.agent_suggestions.agent_suggestions import (
    AgentSuggestions,
)
from generation_service.llm_workflows.tasks.RAG.agent_suggestions.data.input_data import (
    AgentSuggestionsInputParameters,
)
from generation_service.llm_workflows.tasks.RAG.balancer.data.input_data import (
    RagBalancerInputParameters,
)
from generation_service.llm_workflows.tasks.RAG.balancer.rag_balancer import RagBalancer
from generation_service.llm_workflows.tasks.RAG.queries_suggestions.data.input_data import (
    QueriesSuggestionsInputParameters,
)
from generation_service.llm_workflows.tasks.RAG.queries_suggestions.queries_suggestions import (
    QueriesSuggestions,
)
from generation_service.llm_workflows.tasks.augmented_answer_generation.augmented_answer_generation import (
    AugmentedAnswerGeneration,
)
from generation_service.llm_workflows.tasks.augmented_answer_generation.data.input_data import (
    AugmentedGenerationInputParameters,
)
from generation_service.llm_workflows.tasks.call_actions.call_actions import (
    CallActionsExtraction,
)
from generation_service.llm_workflows.tasks.call_reason.call_reason import (
    CallReasonExtraction,
)
from generation_service.llm_workflows.tasks.call_segments.call_segments import (
    CallSegmentsExtraction,
)
from generation_service.llm_workflows.tasks.constrained_object_completion.constrained_object_completion import (
    ConstrainedObjectCompletion,
)
from generation_service.llm_workflows.tasks.constrained_object_completion.data.input_data import (
    ConstrainedCompletionInputParameters,
)

from generation_service.llm_workflows.tasks.contact_reason.contact_reason import (
    ContactReasonExtraction,
)
from generation_service.llm_workflows.tasks.contact_reason_classification.contact_reason_classification import (
    ContactReasonClassification,
)
from generation_service.llm_workflows.tasks.contact_reason_classification.data.input_data import (
    ContactReasonClassificationInputParameters,
)
from generation_service.llm_workflows.tasks.custom_generation.custom_generation import (
    CustomGeneration,
)
from generation_service.llm_workflows.tasks.custom_generation.data.input_data import (
    CustomGenerationInputParameters,
)
from generation_service.llm_workflows.tasks.email_multiclass_tagging.data.input_data import (
    EmailMulticlassTaggingInputParameters,
)
from generation_service.llm_workflows.tasks.email_multiclass_tagging.email_multiclass_tagging import (
    EmailMulticlassTagging,
)
from generation_service.llm_workflows.tasks.follow_up_email.data.input_data import (
    FollowUpEmailInputParameters,
)
from generation_service.llm_workflows.tasks.follow_up_email.follow_up_email import (
    FollowUpEmail,
)
from generation_service.llm_workflows.tasks.knowledge_based_question_answering.data.input_data import (
    KnowledgeBasedQaInputParameters,
)
from generation_service.llm_workflows.tasks.knowledge_based_question_answering.knwowledge_based_question_answering import (
    KnowledgeBasedQuestionAnswering,
)
from generation_service.llm_workflows.tasks.language_detection.language_detection import (
    LanguageDetection,
)
from generation_service.llm_workflows.tasks.llm_evaluation.data.input_data import (
    LlmEvaluationInputParameters,
)
from generation_service.llm_workflows.tasks.llm_evaluation.llm_evaluation import (
    LlmEvaluation,
)
from generation_service.llm_workflows.tasks.named_entity_recognition.data.input_data import (
    NerInputParameters,
)
from generation_service.llm_workflows.tasks.named_entity_recognition.named_entity_recognition import (
    NamedEntityRecognition,
)
from generation_service.llm_workflows.tasks.quality_monitoring.criteria_evaluation.criteria_evaluation import (
    CriteriaEvaluation,
)
from generation_service.llm_workflows.tasks.quality_monitoring.criteria_evaluation.data.input_data import (
    CriteriaEvaluationInputParameters,
)
from generation_service.llm_workflows.tasks.quality_monitoring.criterion_generation.criterion_generation import (
    CriterionGeneration,
)
from generation_service.llm_workflows.tasks.quality_monitoring.criterion_generation.data.input_data import (
    CriterionGenerationInputParameters,
)
from generation_service.llm_workflows.tasks.rating_estimation.data.rating_estimation_input import (
    RatingEstimationInputParameters,
)
from generation_service.llm_workflows.tasks.rating_estimation.rating_estimation import (
    RatingEstimation,
)
from generation_service.llm_workflows.tasks.satisfaction.satisfaction import (
    SatisfactionExtraction,
)
from generation_service.llm_workflows.tasks.satisfaction_factors.data.satisfaction_factors_input import (
    SatisfactionFactorsInputParameters,
)
from generation_service.llm_workflows.tasks.satisfaction_factors.satisfaction_factors import (
    SatisfactionFactorsExtraction,
)
from generation_service.llm_workflows.tasks.summary.summary import CallSummaryExtraction


class TestRunGenerationAction:
    def setup_method(self):
        """
        Setup method to initialize the LLM client and reset mocks for each test.
        This runs automatically before each test method in the class.
        """
        load_dotenv()
        self.lm_client = setup_llm_generation_client(
            base_url="mock_url", api_key="mock_key", asynchronous=True, tokenizer=None
        )
        self.language = AvailableLanguage.ENGLISH

    @pytest.mark.asyncio
    async def test_augmented_answer_generation(self):
        action_id = GenerationAction.AUGMENTED_ANSWER_GENERATION
        action_parameters = {
            "query": "What is the capital of France?",
            "documents": ["Paris is the capital.", "France is in Europe."],
            "context": "Geography quiz",
        }

        with patch.object(
            target=AugmentedAnswerGeneration, attribute="run"
        ) as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=AugmentedGenerationInputParameters(
                query="What is the capital of France?",
                documents=["Paris is the capital.", "France is in Europe."],
                language=self.language,
                context="Geography quiz",
            ),
        )

    @pytest.mark.asyncio
    async def test_call_actions(self):
        action_id = GenerationAction.CALL_ACTIONS
        action_parameters = {
            "transcript": "Customer asked about billing. Agent provided details.",
            "company": "Telecom Inc.",
            "custom_vocabulary": ["billing", "invoice"],
        }

        with patch.object(target=CallActionsExtraction, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=TranscriptBasedInputParameters(
                transcript="Customer asked about billing. Agent provided details.",
                company="Telecom Inc.",
                custom_vocabulary=["billing", "invoice"],
                language=self.language,
            ),
        )

    @pytest.mark.asyncio
    async def test_call_reason(self):
        action_id = GenerationAction.CALL_REASON
        action_parameters = {
            "transcript": "I'm calling about my recent bill.",
            "company": "Utility Co.",
        }

        with patch.object(target=CallReasonExtraction, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=TranscriptBasedInputParameters(
                transcript="I'm calling about my recent bill.",
                company="Utility Co.",
                custom_vocabulary=None,
                language=self.language,
            ),
        )

    @pytest.mark.asyncio
    async def test_call_segments(self):
        action_id = GenerationAction.CALL_SEGMENTS
        action_parameters = {
            "transcript": "I'm calling about my recent bill.",
            "company": "Utility Co.",
        }

        with patch.object(target=CallSegmentsExtraction, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=TranscriptBasedInputParameters(
                transcript="I'm calling about my recent bill.",
                company="Utility Co.",
                custom_vocabulary=None,
                language=self.language,
            ),
        )

    @pytest.mark.asyncio
    async def test_constrained_object_completion(self):
        action_id = GenerationAction.CONSTRAINED_OBJECT_COMPLETION
        action_parameters = {
            "transcript": "Please create a user with name John Doe and email john.doe@example.com",
            "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
            "callback": "my_callback_function",
        }

        with patch.object(
            target=ConstrainedObjectCompletion, attribute="run"
        ) as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=ConstrainedCompletionInputParameters(
                transcript="Please create a user with name John Doe and email john.doe@example.com",
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
                language=self.language,
                callback="my_callback_function",
            ),
        )

    @pytest.mark.asyncio
    async def test_contact_reason(self):
        action_id = GenerationAction.CONTACT_REASON
        action_parameters = {
            "transcript": "I want to change my flight reservation.",
            "company": "FlyAway Airlines",
        }

        with patch.object(target=ContactReasonExtraction, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=TranscriptBasedInputParameters(
                transcript="I want to change my flight reservation.",
                company="FlyAway Airlines",
                custom_vocabulary=None,
                language=self.language,
            ),
        )

    @pytest.mark.asyncio
    async def test_contact_reason_classification(self):
        action_id = GenerationAction.CONTACT_REASON_CLASSIFICATION
        action_parameters = {
            "content": "My internet is not working.",
            "reasons": ["technical issue", "billing", "new service"],
        }

        with patch.object(
            target=ContactReasonClassification, attribute="run"
        ) as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=ContactReasonClassificationInputParameters(
                content="My internet is not working.",
                reasons=["technical issue", "billing", "new service"],
                language=self.language,
            ),
        )

    @pytest.mark.asyncio
    async def test_custom_generation(self):
        action_id = GenerationAction.CUSTOM_GENERATION
        action_parameters = {
            "query": "Write a short poem about nature.",
            "content": "Trees, rivers, mountains, sky.",
            "callback": "process_poem",
        }

        with patch.object(target=CustomGeneration, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=CustomGenerationInputParameters(
                query="Write a short poem about nature.",
                content="Trees, rivers, mountains, sky.",
                language=self.language,
                callback="process_poem",
            ),
        )

    @pytest.mark.asyncio
    async def test_email_multiclass_tagging(self):
        action_id = GenerationAction.EMAIL_MULTICLASS_TAGGING
        action_parameters = {
            "content": "This email is regarding my order #12345.",
            "tags": ["order_inquiry", "shipping_issue", "billing_question"],
            "subject": "Order update",
        }

        with patch.object(target=EmailMulticlassTagging, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=EmailMulticlassTaggingInputParameters(
                content="This email is regarding my order #12345.",
                tags=["order_inquiry", "shipping_issue", "billing_question"],
                use_raw_content=True,
                n_estimators=3,
                subject="Order update",
                language=self.language,
            ),
        )

    @pytest.mark.asyncio
    async def test_follow_up_email(self):
        action_id = GenerationAction.FOLLOW_UP_EMAIL
        action_parameters = {
            "content": "The customer was asking about their refund status.",
        }

        with patch.object(target=FollowUpEmail, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=FollowUpEmailInputParameters(
                content="The customer was asking about their refund status.",
                language=self.language,
            ),
        )

    @pytest.mark.asyncio
    async def test_knowledge_based_question_answering(self):
        action_id = GenerationAction.KNOWLEDGE_BASED_QUESTION_ANSWERING
        action_parameters = {
            "query": "How do I reset my password?",
            "documents": [
                "To reset password, go to settings.",
                "Click on 'Forgot password'.",
            ],
        }

        with patch.object(
            target=KnowledgeBasedQuestionAnswering, attribute="run"
        ) as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=KnowledgeBasedQaInputParameters(
                query="How do I reset my password?",
                documents=[
                    "To reset password, go to settings.",
                    "Click on 'Forgot password'.",
                ],
                language=self.language,
            ),
        )

    @pytest.mark.asyncio
    async def test_language_detection(self):
        action_id = GenerationAction.LANGUAGE_DETECTION
        action_parameters = {"content": "Bonjour, comment ça va?"}

        with patch.object(target=LanguageDetection, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client, content="Bonjour, comment ça va?"
        )

    @pytest.mark.asyncio
    async def test_llm_evaluation(self):
        action_id = GenerationAction.LLM_EVALUATION
        action_parameters = {
            "input": "The quick brown fox.",
            "condition": "Is the sentence grammatically correct?",
            "n_estimators": 5,
            "threshold": 0.8,
        }

        with patch.object(target=LlmEvaluation, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=LlmEvaluationInputParameters(
                input="The quick brown fox.",
                condition="Is the sentence grammatically correct?",
                n_estimators=5,
                threshold=0.8,
            ),
        )

    @pytest.mark.asyncio
    async def test_named_entity_recognition(self):
        action_id = GenerationAction.NAMED_ENTITY_RECOGNITION
        action_parameters = {
            "entities": ["PERSON", "LOCATION"],
            "text": "John Doe lives in New York.",
        }

        with patch.object(target=NamedEntityRecognition, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=NerInputParameters(
                entities=["PERSON", "LOCATION"],
                text="John Doe lives in New York.",
                language=self.language,
            ),
        )

    @pytest.mark.asyncio
    async def test_criteria_evaluation(self):
        action_id = GenerationAction.CRITERIA_EVALUATION
        action_parameters = {
            "criteria": ["politeness", "accuracy"],
            "transcript": "Agent was very polite and answered all questions correctly.",
            "applicability_sampling": 2,
            "evaluation_sampling": 2,
        }

        with patch.object(target=CriteriaEvaluation, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=CriteriaEvaluationInputParameters(
                criteria=["politeness", "accuracy"],
                transcript="Agent was very polite and answered all questions correctly.",
                language=self.language,
                applicability_sampling=2,
                evaluation_sampling=2,
            ),
        )

    @pytest.mark.asyncio
    async def test_criterion_generation(self):
        action_id = GenerationAction.CRITERION_GENERATION
        action_parameters = {
            "title": "Customer Satisfaction",
            "goal": "Improve customer sentiment.",
            "options": ["positive", "negative", "neutral"],
            "applicability_condition": "call duration > 5 min",
            "additional_context": "Focus on resolution.",
        }

        with patch.object(target=CriterionGeneration, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=CriterionGenerationInputParameters(
                title="Customer Satisfaction",
                goal="Improve customer sentiment.",
                options=["positive", "negative", "neutral"],
                applicability_condition="call duration > 5 min",
                additional_context="Focus on resolution.",
                language=self.language,
            ),
        )

    @pytest.mark.asyncio
    async def test_agent_suggestions(self):
        action_id = GenerationAction.AGENT_SUGGESTIONS
        action_parameters = {
            "transcript": "Customer is asking about refund policy.",
            "retrieved_chunks": ["Refunds are processed within 3-5 business days."],
            "suggested_queries": ["refund process", "cancellation policy"],
        }

        with patch.object(target=AgentSuggestions, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=AgentSuggestionsInputParameters(
                transcript="Customer is asking about refund policy.",
                retrieved_chunks=["Refunds are processed within 3-5 business days."],
                suggested_queries=["refund process", "cancellation policy"],
                language=self.language,
            ),
        )

    @pytest.mark.asyncio
    async def test_queries_suggestions(self):
        action_id = GenerationAction.QUERIES_SUGGESTIONS
        action_parameters = {
            "transcript": "I need help with my account.",
            "object_completion": {"account_status": "active"},
            "chat": [{"role": "user", "content": "What is my balance?"}],
            "max_queries": 3,
        }

        with patch.object(target=QueriesSuggestions, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=QueriesSuggestionsInputParameters(
                transcript="I need help with my account.",
                object_completion={"account_status": "active"},
                chat=[{"role": "user", "content": "What is my balance?"}],
                max_queries=3,
                language=self.language,
            ),
        )

    @pytest.mark.asyncio
    async def test_rag_balancer(self):
        action_id = GenerationAction.RAG_BALANCER
        action_parameters = {
            "agent_query": "How do I upgrade my plan?",
            "parallel_evaluations": 2,
            "transcript": "Customer wants to change subscription.",
            "context": "Upgrade options.",
            "chat": [{"role": "user", "content": "Tell me about premium plan."}],
            "documents": ["Premium plan details", "Standard plan details"],
        }

        with patch.object(target=RagBalancer, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=RagBalancerInputParameters(
                agent_query="How do I upgrade my plan?",
                parallel_evaluations=2,
                transcript="Customer wants to change subscription.",
                context="Upgrade options.",
                chat=[{"role": "user", "content": "Tell me about premium plan."}],
                documents=["Premium plan details", "Standard plan details"],
            ),
        )

    @pytest.mark.asyncio
    async def test_rating_estimation(self):
        action_id = GenerationAction.RATING_ESTIMATION
        action_parameters = {
            "review": "This product is amazing! Five stars!",
        }

        with patch.object(target=RatingEstimation, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=RatingEstimationInputParameters(
                review="This product is amazing! Five stars!", language=self.language
            ),
        )

    @pytest.mark.asyncio
    async def test_satisfaction(self):
        action_id = GenerationAction.SATISFACTION
        action_parameters = {
            "transcript": "I am very happy with the service.",
            "company": "Service Provider",
        }

        with patch.object(target=SatisfactionExtraction, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=TranscriptBasedInputParameters(
                transcript="I am very happy with the service.",
                company="Service Provider",
                custom_vocabulary=None,
                language=self.language,
            ),
        )

    @pytest.mark.asyncio
    async def test_satisfaction_factors(self):
        action_id = GenerationAction.SATISFACTION_FACTORS
        action_parameters = {
            "content": "The agent was quick and resolved my issue.",
        }

        with patch.object(
            target=SatisfactionFactorsExtraction, attribute="run"
        ) as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=SatisfactionFactorsInputParameters(
                content="The agent was quick and resolved my issue.",
                language=self.language,
            ),
        )

    @pytest.mark.asyncio
    async def test_summary(self):
        action_id = GenerationAction.SUMMARY
        action_parameters = {
            "transcript": "The customer called to inquire about their internet bill. They had a question about an extra charge. The agent explained the charge was for an on-demand movie and the customer was satisfied.",
            "company": "InternetCo",
            "custom_vocabulary": ["on-demand", "extra charge"],
        }

        with patch.object(target=CallSummaryExtraction, attribute="run") as mock_run:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )

        mock_run.assert_called_once()
        mock_run.assert_called_with(
            lm_client=self.lm_client,
            input_parameters=TranscriptBasedInputParameters(
                transcript="The customer called to inquire about their internet bill. They had a question about an extra charge. The agent explained the charge was for an on-demand movie and the customer was satisfied.",
                company="InternetCo",
                custom_vocabulary=["on-demand", "extra charge"],
                language=self.language,
            ),
        )

    @pytest.mark.asyncio
    async def test_unsupported_action_id(self):
        action_id = "UNSUPPORTED_ACTION"  # type: ignore
        action_parameters = {}

        with pytest.raises(ValueError) as excinfo:
            await run_generation_action(
                lm_client=self.lm_client,
                action_id=action_id,
                action_parameters=action_parameters,
                language=self.language,
            )
        assert f"Unsupported action ID: {action_id}" in str(excinfo.value)
