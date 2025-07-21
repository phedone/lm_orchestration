from typing import Dict, Any

from generation_service.llm_workflows.configuration.clients.baml_client import (
    BamlGenerationClient,
)
from generation_service.llm_workflows.shared.data.available_languages import (
    AvailableLanguage,
)
from generation_service.llm_workflows.shared.data.transcript_based_input_parameters import (
    TranscriptBasedInputParameters,
)
from generation_service.llm_workflows.shared.interfaces.generation_result import (
    GenerationResult,
)
from generation_service.llm_workflows.tasks import GenerationAction
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


async def run_generation_action(
    lm_client: BamlGenerationClient,
    action_id: GenerationAction,
    action_parameters: Dict[str, Any],
    language: AvailableLanguage,
) -> GenerationResult:
    match action_id:
        case GenerationAction.AUGMENTED_ANSWER_GENERATION:
            input_parameters = AugmentedGenerationInputParameters(
                query=action_parameters["query"],
                documents=action_parameters["documents"],
                language=language,
                context=action_parameters.get("context", None),
            )
            return await AugmentedAnswerGeneration.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.CALL_ACTIONS:
            input_parameters = TranscriptBasedInputParameters(
                transcript=action_parameters["transcript"],
                company=action_parameters.get("company"),
                custom_vocabulary=action_parameters.get("custom_vocabulary"),
                language=language,
            )

            return await CallActionsExtraction.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.CALL_REASON:
            input_parameters = TranscriptBasedInputParameters(
                transcript=action_parameters["transcript"],
                company=action_parameters.get("company"),
                custom_vocabulary=action_parameters.get("custom_vocabulary"),
                language=language,
            )

            return await CallReasonExtraction.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.CALL_SEGMENTS:
            input_parameters = TranscriptBasedInputParameters(
                transcript=action_parameters["transcript"],
                company=action_parameters.get("company"),
                custom_vocabulary=action_parameters.get("custom_vocabulary"),
                language=language,
            )

            return await CallSegmentsExtraction.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.CONSTRAINED_OBJECT_COMPLETION:
            input_parameters = ConstrainedCompletionInputParameters(
                transcript=action_parameters["transcript"],
                schema=action_parameters["schema"],
                language=language,
                callback=action_parameters.get("callback", None),
            )

            return await ConstrainedObjectCompletion.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.CONTACT_REASON:
            input_parameters = TranscriptBasedInputParameters(
                transcript=action_parameters["transcript"],
                company=action_parameters.get("company"),
                custom_vocabulary=action_parameters.get("custom_vocabulary"),
                language=language,
            )

            return await ContactReasonExtraction.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.CONTACT_REASON_CLASSIFICATION:
            input_parameters = ContactReasonClassificationInputParameters(
                content=action_parameters["content"],
                reasons=action_parameters["reasons"],
                language=language,
            )

            return await ContactReasonClassification.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.CUSTOM_GENERATION:
            input_parameters = CustomGenerationInputParameters(
                query=action_parameters["query"],
                content=action_parameters.get("content", None),
                language=language,
                callback=action_parameters.get("callback", None),
            )

            return await CustomGeneration.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.EMAIL_MULTICLASS_TAGGING:
            input_parameters = EmailMulticlassTaggingInputParameters(
                content=action_parameters["content"],
                tags=action_parameters["tags"],
                use_raw_content=action_parameters.get("use_raw_content", True),
                n_estimators=action_parameters.get("n_estimators", 3),
                subject=action_parameters.get("subject", None),
                language=language,
            )

            return await EmailMulticlassTagging.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.FOLLOW_UP_EMAIL:
            input_parameters = FollowUpEmailInputParameters(
                content=action_parameters["content"],
                language=language,
            )

            return await FollowUpEmail.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.KNOWLEDGE_BASED_QUESTION_ANSWERING:
            input_parameters = KnowledgeBasedQaInputParameters(
                query=action_parameters["query"],
                documents=action_parameters["documents"],
                language=language,
            )

            return await KnowledgeBasedQuestionAnswering.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.LANGUAGE_DETECTION:
            return await LanguageDetection.run(
                lm_client=lm_client,
                content=action_parameters["content"],
            )

        case GenerationAction.LLM_EVALUATION:
            input_parameters = LlmEvaluationInputParameters(
                input=action_parameters["input"],
                condition=action_parameters["condition"],
                n_estimators=action_parameters.get("n_estimators", 10),
                threshold=action_parameters.get("threshold", 0.5),
            )

            return await LlmEvaluation.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.NAMED_ENTITY_RECOGNITION:
            input_parameters = NerInputParameters(
                entities=action_parameters["entities"],
                text=action_parameters["text"],
                language=language,
            )

            return await NamedEntityRecognition.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.CRITERIA_EVALUATION:
            input_parameters = CriteriaEvaluationInputParameters(
                criteria=action_parameters["criteria"],
                transcript=action_parameters["transcript"],
                language=language,
                applicability_sampling=action_parameters.get(
                    "applicability_sampling", 3
                ),
                evaluation_sampling=action_parameters.get("evaluation_sampling", 3),
            )

            return await CriteriaEvaluation.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.CRITERION_GENERATION:
            # non streamed version here
            input_parameters = CriterionGenerationInputParameters(
                title=action_parameters["title"],
                goal=action_parameters["goal"],
                options=action_parameters["options"],
                applicability_condition=action_parameters.get(
                    "applicability_condition", None
                ),
                additional_context=action_parameters.get("additional_context", None),
                language=language,
            )

            return await CriterionGeneration.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.AGENT_SUGGESTIONS:
            input_parameters = AgentSuggestionsInputParameters(
                transcript=action_parameters["transcript"],
                retrieved_chunks=action_parameters["retrieved_chunks"],
                suggested_queries=action_parameters["suggested_queries"],
                language=language,
            )

            return await AgentSuggestions.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.QUERIES_SUGGESTIONS:
            input_parameters = QueriesSuggestionsInputParameters(
                transcript=action_parameters["transcript"],
                object_completion=action_parameters.get("object_completion", None),
                chat=action_parameters.get("chat", None),
                max_queries=action_parameters.get("max_queries", 5),
                language=language,
            )

            return await QueriesSuggestions.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.RAG_BALANCER:
            input_parameters = RagBalancerInputParameters(
                agent_query=action_parameters["agent_query"],
                parallel_evaluations=action_parameters.get("parallel_evaluations", 3),
                transcript=action_parameters.get("transcript", None),
                context=action_parameters.get("context", None),
                chat=action_parameters.get("chat", None),
                documents=action_parameters.get("documents", None),
            )

            return await RagBalancer.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.RATING_ESTIMATION:
            input_parameters = RatingEstimationInputParameters(
                review=action_parameters["review"],
                language=language,
            )

            return await RatingEstimation.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.SATISFACTION:
            input_parameters = TranscriptBasedInputParameters(
                transcript=action_parameters["transcript"],
                company=action_parameters.get("company"),
                custom_vocabulary=action_parameters.get("custom_vocabulary"),
                language=language,
            )

            return await SatisfactionExtraction.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.SATISFACTION_FACTORS:
            input_parameters = SatisfactionFactorsInputParameters(
                content=action_parameters["content"],
                language=language,
            )

            return await SatisfactionFactorsExtraction.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case GenerationAction.SUMMARY:
            input_parameters = TranscriptBasedInputParameters(
                transcript=action_parameters["transcript"],
                company=action_parameters.get("company"),
                custom_vocabulary=action_parameters.get("custom_vocabulary"),
                language=language,
            )

            return await CallSummaryExtraction.run(
                lm_client=lm_client, input_parameters=input_parameters
            )

        case _:
            raise ValueError(f"Unsupported action ID: {action_id}")
