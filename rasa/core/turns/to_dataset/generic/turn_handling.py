from dataclasses import dataclass
from typing import Generic, List, Optional

from rasa.core.turns.turn import Actor
from rasa.core.turns.to_dataset.turn_sequences import (
    TurnSequenceValidation,
    TurnSequenceModifier,
    TurnType,
)
from rasa.shared.core.events import ActionExecuted


class EndsWithBotTurn(TurnSequenceValidation[TurnType]):
    def validate(self, turns: List[TurnType]) -> bool:
        return turns[-1].actor == Actor.BOT


@dataclass
class HasMinLength(TurnSequenceValidation[TurnType]):
    min_length: int

    def validate(self, turns: List[TurnType]) -> bool:
        return len(turns) >= self.min_length


@dataclass
class EndsWithPredictableActionExecuted(TurnSequenceValidation[TurnType]):
    def validate(self, turns: List[TurnType]) -> bool:
        if not turns[-1].events:
            return False
        first_event_of_last_turn = turns[-1].events[0]
        return (
            isinstance(first_event_of_last_turn, ActionExecuted)
            and not first_event_of_last_turn.unpredictable  # TODO: this should not be
            # needed
        )


@dataclass
class RemoveLastTurn(TurnSequenceModifier[TurnType], Generic[TurnType]):
    def modfify(
        self, turns: List[TurnType], training: bool, inplace_allowed: bool
    ) -> List[TurnType]:
        return turns[:-1]


@dataclass
class KeepMaxHistory(TurnSequenceModifier[TurnType], Generic[TurnType]):

    max_history: Optional[int] = None
    offset_for_training: int = 0

    def modify(
        self, turns: List[TurnType], training: bool, inplace_allowed: bool
    ) -> List[TurnType]:
        """Keeps the last `max_history`(+1) turns during inference (training)."""
        if self.max_history is not None:
            keep = (
                (self.max_history + self.offset_for_training)
                if training
                else self.max_history
            )
            turns = turns[-keep:]
        return turns
