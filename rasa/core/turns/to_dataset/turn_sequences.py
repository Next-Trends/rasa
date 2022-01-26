from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Generic, Optional, List, TypeVar, Dict, Iterator, Set, Tuple, Any

from rasa.shared.core.domain import Domain
from rasa.core.turns.to_dataset.utils.trainable import Trainable
from rasa.core.turns.turn import Turn
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.training_data.features import Features


logger = logging.getLogger(__name__)

TurnType = TypeVar("TurnType")


def steps2str(steps: List[Any]) -> str:
    indent = " " * 2
    steps = "\n".join(f"{indent}{idx:2}. {turn}" for idx, turn in enumerate(steps))
    return f"[\n{steps}\n]"


class TurnSequenceGenerator(Generic[TurnType], ABC):
    """Generates sub-sequences from a given sequence of turns."""

    def __init__(
        self,
        preprocessing: Optional[List[TurnSequenceModifier[TurnType]]],
        filters: Optional[List[TurnSequenceValidation[TurnType]]],
        ignore_duplicates: bool,
        modifiers: Optional[List[TurnSequenceModifier[TurnType]]],
        filter_results: bool,
    ) -> None:
        """
        Args:
            filters: only applied during training
            ...
        """
        self._preprocessing = preprocessing or []
        self._filters = filters or []
        self._ignore_duplicates = ignore_duplicates
        self._cache: Set[Tuple[int, ...]] = set()
        self._modifiers = modifiers or []
        self._filter_results = filter_results

    def apply_to(
        self, turns: List[TurnType], training: bool, limit: Optional[int] = None,
    ) -> Iterator[List[Turn]]:
        """

        During training, the whole sequence of turns is processed.

        During validation, we extract multiple (or no) sub-sequences of turns from the
        given sequence of turns: Each subsequence of turns from the 0-th to the i-th
        turn that passes the given sequence filters and is not a duplicate of any
        other subsequence created so far.

        In both cases, the same modifiers are applied.

        Args:
            turns: ...
            training: ...
            limit: ...

        """
        steps = [len(turns) + 1] if not training else range(2, len(turns) + 1)
        num_generated = 0

        logger.debug(f"Start generating subsequences from:\n{steps2str(turns)}")

        preprocessed_turns = TurnSequenceModifier.apply_all(
            modifiers=self._preprocessing,
            turns=turns,
            training=True,
            inplace_allowed=(not training),
        )

        logger.debug(f"Applied pre-processing:\n{steps2str(preprocessed_turns)}")

        for idx in steps:

            if limit and num_generated >= limit:
                return

            # we'll make a copy of this subsequence, once we know we continue with it
            subsequence = preprocessed_turns[:idx]

            logger.debug(
                f"Attempt to generate from subsequence:\n{steps2str(subsequence)}"
            )

            # during training - skip if it does not pass filters
            if training and not TurnSequenceValidation.apply_all(
                validations=self._filters, turns=subsequence
            ):
                logger.debug(f"Failed (did not pass filters {self._filters})")
                continue

            # apply modifiers
            subsequence = TurnSequenceModifier.apply_all(
                modifiers=self._modifiers,
                turns=subsequence,
                training=training,
                inplace_allowed=(not training),
            )

            if self._modifiers:
                logger.debug(f"Modified subsequence:\n{steps2str(subsequence)}")

            # during training - check if filters still pass (we modified the sequence)
            if training and not TurnSequenceValidation.apply_all(
                self._filters, subsequence
            ):
                logger.debug(f"Failed (did not pass filters {self._filters})")
                continue

            # during training - skip if it is a duplicate
            if training and self._ignore_duplicates:
                id = tuple(hash(turn) for turn in subsequence)
                if id in self._cache:
                    logger.debug(f"Failed (duplicate of other subsequence)")
                    continue
                else:
                    self._cache.add(id)

            num_generated += 1
            yield subsequence


@dataclass
class TurnSequenceValidation(Generic[TurnType]):
    """Determines whether or not a given list of turns satisfies some criteria."""

    @abstractmethod
    def validate(self, turns: List[TurnType]) -> bool:
        raise NotImplementedError

    @staticmethod
    def apply_all(
        validations: List[TurnSequenceValidation[TurnType]], turns: List[TurnType]
    ) -> bool:
        return all(validation.validate(turns) for validation in validations)


@dataclass
class TurnSequenceModifier(Generic[TurnType], ABC):
    """Returns a modified list of turns.

    Must not modify the given list of turns.
    """

    on_training: bool = True
    on_inference: bool = True

    @abstractmethod
    def modify(
        self, turns: List[TurnType], training: bool, inplace_allowed: bool
    ) -> List[TurnType]:
        """Returns a modified turn sequence.

        Args:
            turns: a list of turns
            training: whether or not we're in training mode
            inplace_allowed: if this is set to `False` then the single turns of the given
               sequence of turns may be modified inplace_allowed; otherwise the given turns
               must not be modified but the returned turn sequence may contain new
               turn objects
        Returns:
            a modified turn sequence
        """
        raise NotImplementedError

    def apply_to(
        self, turns: List[TurnType], inplace_allowed: bool, training: bool,
    ) -> List[TurnType]:
        if (self.on_training and training) or (self.on_inference and not training):
            return self.modify(
                turns=turns, training=training, inplace_allowed=inplace_allowed,
            )
        return turns

    @staticmethod
    def apply_all(
        modifiers: List[TurnSequenceModifier[TurnType]],
        turns: List[TurnType],
        training: bool,
        inplace_allowed: bool,
    ) -> List[TurnType]:
        for modifier in modifiers:
            turns = modifier.apply_to(
                turns, inplace_allowed=inplace_allowed, training=training
            )
        return turns


class TurnFeaturizer(Generic[TurnType], Trainable, ABC):
    """Featurize a single turn."""

    def train(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        self._train(domain=domain, interpreter=interpreter)
        self._trained = True

    @abstractmethod
    def _train(self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]):
        raise NotImplementedError

    @abstractmethod
    def featurize(
        self,
        turn: TurnType,
        interpreter: NaturalLanguageInterpreter,
        training: bool = True,
    ) -> Dict[str, List[Features]]:
        raise NotImplementedError
