from __future__ import annotations
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Text,
    Generic,
    Tuple,
    Optional,
)
import logging

import numpy as np

from rasa.core.turns.to_dataset.labels import (
    LabelFromTurnsExtractor,
    LabelFeaturizationPipeline,
)
from rasa.core.turns.turn import Turn
from rasa.core.turns.to_dataset.turn_sequences import (
    TurnSequenceGenerator,
    TurnType,
    steps2str,
    TurnFeaturizer,
)
from rasa.core.turns.to_dataset.utils.trainable import Trainable
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.training_data.features import Features

logger = logging.getLogger(__name__)

FeatureCollection = Dict[str, List[Features]]


class DatasetGenerator(Generic[TurnType]):
    """Generates labeled and modified subsequences of turns."""

    def __init__(
        self,
        turn_sequence_generator: TurnSequenceGenerator[TurnType],
        label_extractors: Dict[Text, LabelFromTurnsExtractor[TurnType, Any]],
    ):
        self._turn_sequence_generator = turn_sequence_generator
        self._label_extractors = label_extractors

    def apply_to(
        self, turns: List[TurnType], training: bool,
    ) -> Iterator[Tuple[List[Turn], Optional[Dict[Text, Any]]]]:
        for processed_turns in self._turn_sequence_generator.apply_to(
            turns, training=training
        ):
            processed_turns, outputs = LabelFromTurnsExtractor.apply_all(
                self._label_extractors,
                turns=processed_turns,
                training=training,
                inplace_allowed=(not training),
            )
            # logger.debug(f"Generated: \n{turns_to_str(processed_turns)}") # TODO:
            yield processed_turns, outputs


class FeaturizedDatasetGenerator(Generic[TurnType], Trainable):
    """Generates and encodes labeled and modified subsequences of turns."""

    # TODO: add __len__ implementation that skips the featurization

    def __init__(
        self,
        turn_sequence_generator: TurnSequenceGenerator[TurnType],
        label_pipelines: LabelFeaturizationPipeline[TurnType, Any],
        turn_featurizer: TurnFeaturizer[TurnType],
    ):
        self._turn_sequence_generator = turn_sequence_generator
        self._label_pipelines = label_pipelines
        self._turn_featurizer = turn_featurizer

    def train_featurizers_and_indexers(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        self._turn_featurizer.train(domain=domain, interpreter=interpreter)
        for pipeline in self._label_pipelines:
            if pipeline.featurizer:
                pipeline.featurizer.train(domain=domain, interpreter=interpreter)
            if pipeline.indexer:
                pipeline.indexer.train(domain=domain, extractor=pipeline.extractor)
        self._trained = True

    def apply_to(
        self,
        turns: List[TurnType],
        interpreter: NaturalLanguageInterpreter,
        training: bool,
    ) -> Iterator[
        Tuple[List[FeatureCollection], FeatureCollection, Dict[Text, np.ndarray]]
    ]:
        self.raise_if_not_trained()

        logger.debug("=" * 100)
        for subsequence in self._turn_sequence_generator.apply_to(
            turns, training=training
        ):

            logger.debug(
                f"<<1>>. Extracting next sequence from:\n{steps2str(subsequence)}"
            )

            # extract and featurize labels during training
            collected_features = {}
            collected_indices = {}
            subsequence_modified = subsequence
            if training:

                for label_pipeline in self._label_pipelines:
                    label_name = label_pipeline.name
                    (
                        subsequence_modified,
                        next_features,
                        next_indices,
                    ) = label_pipeline.apply_to(
                        subsequence_modified,
                        training=training,
                        inplace_allowed=(not training),
                        interpreter=interpreter,
                    )  # TODO: return raw labels as well for debugging
                    collected_features[label_name] = next_features
                    collected_indices[label_name] = next_indices
                logger.debug(
                    f"<<2>>. Sequence after label handling:\n"
                    f"{steps2str(subsequence)}"
                )
                logger.debug(f"<<3>>. Collected features : {collected_features}")
                logger.debug(f"<<4>>. Collected indices : {collected_features}")

            # featurize the (remaining) input (during training)
            subsequence_featurized = [
                self._turn_featurizer.featurize(turn, interpreter=interpreter)
                for turn in subsequence_modified
            ]
            logger.debug(
                f"<<5>>. After featurization:\n" f"{steps2str(subsequence_featurized)}"
            )
            logger.debug("-" * 100)
            yield subsequence_featurized, collected_features, collected_indices
