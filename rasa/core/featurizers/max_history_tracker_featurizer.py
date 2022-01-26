from typing import Tuple, List, Optional, Dict, Text, Any
import numpy as np


from rasa.core.turns.stateful.stateful_turn import StatefulTurn
from rasa.core.turns.stateful.stateful_turn_featurizers import (
    BasicStatefulTurnFeaturizer,
)
from rasa.core.turns.stateful.stateful_turn_labels import (
    ExtractActionFromLastTurn,
    ExtractEntitiesFromLastUserTurn,
)
from rasa.core.turns.stateful.stateful_turn_handling import (
    IfLastTurnWasUserTurnKeepEitherTextOrNonText,
    RemoveTurnsWithPrevActionUnlikelyIntent,
    RemoveUserTextIfIntentFromEveryTurn,
)
from rasa.core.turns.to_dataset.generic.label_handling import (
    ApplyEntityTagsEncoder,
    IndexerFromLabelExtractor,
)
from rasa.core.turns.to_dataset.generic.turn_handling import (
    EndsWithPredictableActionExecuted,
    KeepMaxHistory,
)
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.turns.to_dataset.dataset import (
    DatasetGenerator,
    FeaturizedDatasetGenerator,
)
from rasa.core.turns.to_dataset.labels import LabelFeaturizationPipeline
from rasa.core.turns.to_dataset.turn_sequences import (
    TurnSequenceGenerator,
    TurnSequenceModifier,
)
from rasa.shared.core.domain import State, Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.constants import ACTION_NAME, ENTITY_TAGS
from rasa.shared.nlu.training_data.features import Features


# from rasa.utils.tensorflow.constants import LABEL_PAD_ID


from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer


class MaxHistoryTrackerFeaturizer(TrackerFeaturizer):

    LABEL_NAME = "action"

    def __init__(
        self,
        state_featurizer: Optional[SingleStateFeaturizer] = None,
        max_history: Optional[int] = None,
        remove_duplicates: bool = True,
    ) -> None:
        super().__init__(state_featurizer)
        self.max_history = max_history
        self.remove_duplicates = remove_duplicates

    def prepare_for_featurization(
        self,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        bilou_tagging: bool = False,
    ) -> None:
        """

        preparation for create_state_features (inference) and
        featurize_trackers (training) - used by ML policies

        """

        # the following is an ugly workaround to be able to set dynamically -- which
        # should not be necessary ...
        self._featurized_dataset_generator_ignore_action_unlikely = RemoveTurnsWithPrevActionUnlikelyIntent(
            switched_on=False
        )
        turn_sequence_generator = TurnSequenceGenerator(
            preprocessing=[
                RemoveUserTextIfIntentFromEveryTurn(),
                self._featurized_dataset_generator_ignore_action_unlikely,
            ],
            filters=[EndsWithPredictableActionExecuted()],
            ignore_duplicates=self.remove_duplicates,
            modifiers=[
                KeepMaxHistory(max_history=self.max_history, offset_for_training=+1)
            ],
            filter_results=None,
        )
        label_pipelines = (
            LabelFeaturizationPipeline(
                name=ACTION_NAME,
                extractor=ExtractActionFromLastTurn(remove_last_turn=True),
                featurizer=None,
                indexer=IndexerFromLabelExtractor(),
            ),
            # (
            #     INTENT,
            #     LabelFeaturizationPipeline(
            #         extractor=ExtractIntentFromLastUserTurn(),
            #         featurizer=LabelFeaturizerViaLookup(attribute=INTENT),
            #         indexer=IndexerFromLabelExtractor(),
            #     ),
            # ),
            LabelFeaturizationPipeline(
                name=ENTITY_TAGS,
                extractor=ExtractEntitiesFromLastUserTurn(),
                featurizer=ApplyEntityTagsEncoder(bilou_tagging=bilou_tagging),
                indexer=None,  # this won't work
            ),
        )
        self._featurized_dataset_generator = FeaturizedDatasetGenerator(
            turn_sequence_generator=turn_sequence_generator,
            label_pipelines=label_pipelines,
            turn_featurizer=BasicStatefulTurnFeaturizer(),
        )

        self._featurized_dataset_generator.train_featurizers_and_indexers(
            domain=domain, interpreter=interpreter
        )

    def featurize_trackers(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        bilou_tagging: bool = False,
        ignore_action_unlikely_intent: bool = False,
    ) -> Tuple[
        List[List[Dict[Text, List[Features]]]],
        np.ndarray,
        List[List[Dict[Text, List[Features]]]],
    ]:
        """

        used by ML policies during training

        """

        # FIXME: was is this done here *and* public - if so, why?
        self.prepare_for_featurization(
            domain=domain, interpreter=interpreter, bilou_tagging=bilou_tagging
        )

        trackers_as_states = []
        trackers_as_labels = []
        trackers_as_entities = []

        # TODO: do we need this flexibility?
        self._featurized_dataset_generator_ignore_action_unlikely.switched_on = (
            ignore_action_unlikely_intent
        )

        for tracker in trackers:

            stateful_turns = StatefulTurn.parse(
                tracker=tracker,
                domain=domain,
                omit_unset_slots=False,
                ignore_rule_only_turns=False,  # no need to during training..
                rule_only_data=False,  # no need to during training..
            )

            print("stateful_turns: ", stateful_turns)

            for (
                inputs,
                label_features,
                label_indices,
            ) in self._featurized_dataset_generator.apply_to(
                turns=stateful_turns, interpreter=interpreter, training=True
            ):

                trackers_as_states.append(inputs)
                trackers_as_labels.append(label_indices[ACTION_NAME])
                trackers_as_entities.append(label_features[ENTITY_TAGS])

                print("trackers_as_states: ", inputs)

        return trackers_as_states, np.array(trackers_as_labels), trackers_as_entities

    def create_state_features(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        use_text_for_last_user_input: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        ignore_action_unlikely_intent: bool = False,
    ) -> List[List[Dict[Text, List[Features]]]]:
        """

        used by ML policies during inference

        """
        pass

    def training_states_and_labels(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        omit_unset_slots: bool = False,
        ignore_action_unlikely_intent: bool = False,
    ) -> Tuple[List[List[State]], List[List[Text]]]:
        """

        used by rule policies during training (no unit tests for this)

        """
        pass

    def _for_prediction_states(
        self,
    ) -> Tuple[DatasetGenerator, TurnSequenceModifier, TurnSequenceModifier]:
        # the following is an ugly workaround to be able to set dynamically -- which
        # should not be necessary ...
        ignore_action_unlikely = RemoveTurnsWithPrevActionUnlikelyIntent(
            switched_on=True
        )
        last_turn_handling = IfLastTurnWasUserTurnKeepEitherTextOrNonText(
            on_training=False, keep_text=True
        )
        modifiers = [
            ignore_action_unlikely,
            KeepMaxHistory(max_history=self.max_history),
            last_turn_handling,
            RemoveUserTextIfIntentFromEveryTurn(on_training=False),
        ]
        turn_sequence_generator = TurnSequenceGenerator(
            preprocessing=None,
            filters=None,
            ignore_duplicates=self.remove_duplicates,
            modifiers=modifiers,
            filter_results=None,
        )
        dataset = DatasetGenerator(
            turn_sequence_generator=turn_sequence_generator, label_extractors={},
        )
        return dataset, ignore_action_unlikely, last_turn_handling

    def prediction_states(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        use_text_for_last_user_input: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        ignore_action_unlikely_intent: bool = False,
    ) -> List[List[State]]:
        """

        used by rule-policies during inference

        """

        # Note: The generator only needs to be setup once -- but I'm lazy atm
        (
            generator,
            ignore_action_unlikely,
            last_turn_handling,
        ) = self._for_prediction_states()

        # TODO: is this really needed? - these should be fixed per policy
        ignore_action_unlikely.switched_on = ignore_action_unlikely_intent
        last_turn_handling.keep_text = use_text_for_last_user_input

        trackers_as_states = []
        for tracker in trackers:

            stateful_turns = StatefulTurn.parse(
                tracker=tracker,
                domain=domain,
                omit_unset_slots=False,
                ignore_rule_only_turns=ignore_rule_only_turns,
                rule_only_data=False,
            )

            modified_turns, _ = next(generator.apply_to(stateful_turns, training=False))

            trackers_as_states.append([turn.state for turn in modified_turns])

        return trackers_as_states
