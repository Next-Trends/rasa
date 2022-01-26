from __future__ import annotations
from abc import abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Text,
    TypeVar,
    Generic,
    Tuple,
    Optional,
)
from dataclasses import dataclass
import logging

import numpy as np

from rasa.core.turns.to_dataset.turn_sequences import TurnType, steps2str
from rasa.core.turns.to_dataset.utils.trainable import Trainable
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.training_data.features import Features

logger = logging.getLogger(__name__)

RawLabelType = TypeVar("RawLabelType")


@dataclass
class LabelFeaturizationPipeline(Generic[TurnType, RawLabelType]):
    """Extracts labels, featurizes them and converts them to labels."""

    name: str
    extractor: LabelFromTurnsExtractor[TurnType, RawLabelType]
    featurizer: Optional[LabelFeaturizer[RawLabelType]]
    indexer: Optional[LabelIndexer[TurnType, RawLabelType]]

    def apply_to(
        self,
        turns: List[TurnType],
        training: bool,
        inplace_allowed: bool,
        interpreter: Optional[NaturalLanguageInterpreter] = None,
    ) -> Tuple[List[TurnType], List[Features], np.ndarray]:
        turns, extracted_label = self.extractor.apply_to(
            turns=turns, training=training, inplace_allowed=inplace_allowed
        )
        features = []
        indices = np.array([])
        assert (extracted_label is not None) == training
        if training:
            if self.featurizer:
                features = self.featurizer.featurize(
                    raw_label=extracted_label, interpreter=interpreter
                )
            if self.indexer:
                indices = self.indexer.index(raw_label=extracted_label)
        return turns, features, indices


class LabelFromTurnsExtractor(Generic[TurnType, RawLabelType]):
    """Extracts label information from a sequence of turns."""

    on_training: bool = True
    on_inference: bool = False

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def extract(
        self, turns: List[TurnType], training: bool, inplace_allowed: bool
    ) -> Tuple[List[TurnType], RawLabelType]:
        raise NotImplementedError

    @abstractmethod
    def apply_to(
        self, turns: List[TurnType], training: bool, inplace_allowed: bool
    ) -> Tuple[List[TurnType], RawLabelType]:
        if (training and self.on_training) or (not training and self.on_inference):
            turns, raw_label = self.extract(
                turns, training=training, inplace_allowed=inplace_allowed
            )
            logger.debug(f"{self.__class__.__name__} extracted: {raw_label}")
            logger.debug(f"Remaining turns:\n{steps2str(turns)}")
            return turns, raw_label
        return turns, None

    def from_domain(self, domain: Domain) -> List[RawLabelType]:
        # TODO: do we need to be able to handle a new domain here?
        raise NotImplementedError

    @classmethod
    def apply_all(
        cls,
        label_extractors: List[Tuple[Text, LabelFromTurnsExtractor[TurnType, Any]]],
        turns: List[TurnType],
        training: bool,
        inplace_allowed: bool,
    ) -> Dict[Text, Any]:
        outputs = {}
        for name, extractor in label_extractors:
            turns, extracted = extractor.apply_to(
                turns=turns, training=training, inplace_allowed=inplace_allowed
            )
            if extracted:
                outputs[name] = extracted
        return turns, outputs


class LabelFeaturizer(Generic[RawLabelType], Trainable):
    """Converts a label to `Features`."""

    def featurize(
        self, raw_label: RawLabelType, interpreter: NaturalLanguageInterpreter
    ) -> List[Features]:
        self.raise_if_not_trained()
        return self._featurize(raw_label=raw_label, interpreter=interpreter)

    @abstractmethod
    def _featurize(
        self, raw_label: RawLabelType, interpreter: NaturalLanguageInterpreter
    ) -> List[Features]:
        raise NotImplementedError

    def train(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        self._train(domain=domain, interpreter=interpreter)
        self._trained = True

    @abstractmethod
    def _train(self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]):
        raise NotImplementedError


class LabelIndexer(Generic[TurnType, RawLabelType], Trainable):
    """Converts a label to an index."""

    def index(self, raw_label: Optional[RawLabelType],) -> np.ndarray:
        self.raise_if_not_trained()
        return self._index(raw_label=raw_label)

    @abstractmethod
    def _index(self, raw_label: Optional[RawLabelType]) -> np.ndarray:
        raise NotImplementedError

    def train(
        self, domain: Domain, extractor: LabelFromTurnsExtractor[TurnType, RawLabelType]
    ) -> None:
        self._train(domain=domain, extractor=extractor)
        self._trained = True

    @abstractmethod
    def _train(
        self, domain: Domain, extractor: LabelFromTurnsExtractor[TurnType, RawLabelType]
    ) -> None:
        raise NotImplementedError
