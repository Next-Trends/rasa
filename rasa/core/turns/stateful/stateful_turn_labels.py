from __future__ import annotations

import copy
from dataclasses import dataclass
from abc import ABC
from typing import (
    Any,
    List,
    Optional,
    Dict,
    Text,
    Tuple,
    TypeVar,
)

from rasa.core.turns.turn import Turn
from rasa.core.turns.stateful.stateful_turn import StatefulTurn
from rasa.core.turns.to_dataset.dataset import LabelFromTurnsExtractor
from rasa.shared.core.constants import PREVIOUS_ACTION, USER
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.constants import (
    ACTION_NAME,
    ACTION_TEXT,
    ENTITIES,
    INTENT,
    TEXT,
)


ACTION_NAME_OR_TEXT = "action_name_or_text"

AttributeType = TypeVar("AttributeType")


class ExtractAttributeFromLastUserTurn(
    LabelFromTurnsExtractor[StatefulTurn, Tuple[Optional[Text], AttributeType]], ABC
):
    """Extracts an attribute from the user-substate of the last user turn.

    The information will be removed from the user sub-state of the last user turn
    and of all following turns.

    Along with the attribute, the user text will be returned. However, unlike the
    extracted attribute, the user text will **not** be removed from any substate.
    """

    def __init__(self, attribute: Text) -> None:
        self._attribute = attribute

    def extract(
        self, turns: List[StatefulTurn], training: bool, inplace_allowed: bool
    ) -> Tuple[List[Turn], Tuple[Optional[Text], Optional[AttributeType]]]:
        last_user_turn_idx = Turn.get_index_of_last_user_turn(turns)
        if last_user_turn_idx is None:
            return turns, (None, None)
        last_user_turn = turns[last_user_turn_idx]
        state = last_user_turn.state.get(USER, {})
        raw_info: Tuple[Optional[Text], Optional[AttributeType]] = (
            last_user_turn.state.get(TEXT, None),
            state.get(self._attribute, None),
        )
        # it is the last user turn, but the states in all subsequent bot turns
        # contain information about the last user turn
        for idx in range(last_user_turn_idx, len(turns)):
            if not inplace_allowed:
                turns[idx] = copy.deepcopy(turns[idx])
            turns[idx].state.get(USER, {}).pop(self._attribute, None)
        return turns, raw_info

    def from_domain(self, domain: Domain,) -> List[Tuple[Optional[Text], AttributeType]]:
        raise NotImplementedError()


class ExtractIntentFromLastUserTurn(LabelFromTurnsExtractor[StatefulTurn, Text]):
    """Extract the intent from the last user turn.

    The intent will be removed from the user sub-state of the last user turn and of
    all following turns.
    """

    def __init__(self) -> None:
        self.extractor = ExtractAttributeFromLastUserTurn(attribute=INTENT)

    def extract(
        self, turns: List[StatefulTurn], training: bool, inplace_allowed: bool
    ) -> Tuple[List[Turn], Tuple[Optional[Text], Optional[Text]]]:
        turns, (_, intent) = self.extractor.extract(
            turns=turns, training=training, inplace_allowed=inplace_allowed
        )
        return turns, intent

    def from_domain(self, domain: Domain,) -> List[str]:
        return domain.intents


class ExtractEntitiesFromLastUserTurn(
    LabelFromTurnsExtractor[StatefulTurn, Optional[Dict[Text, Any]]]
):
    """Extract the entities from the last user turn.

    The entities will be removed from the user sub-state of the
    last user turn and of all following turns.
    """

    def __init__(self) -> None:
        self.extractor = ExtractAttributeFromLastUserTurn(attribute=ENTITIES)

    def extract(
        self, turns: List[StatefulTurn], training: bool, inplace_allowed: bool
    ) -> Tuple[List[Turn], Optional[Dict[Text, Any]]]:
        turns, (text, entities) = self.extractor.extract(
            turns=turns, training=training, inplace_allowed=inplace_allowed
        )
        return turns, (text, entities)

    def from_domain(self, domain: Domain,) -> List[Text]:
        raise NotImplementedError()  # TODO get list of entities from domain


@dataclass
class ExtractActionFromLastTurn(LabelFromTurnsExtractor[StatefulTurn, Text]):
    """Extracts the action from the last turn.

    If an action name or an action text exist, those will be returned (with action
    names being chosen over action text - if both exist).
    The complete action sub-state will be removed from the last turn afterwards.

    Args:
        remove_last_turn: set to True to remove the last turn completely
    """

    remove_last_turn: bool = False

    def extract(
        self, turns: List[StatefulTurn], training: bool, inplace_allowed: bool
    ) -> Tuple[List[StatefulTurn], Text]:
        prev_action = turns[-1].state.get(PREVIOUS_ACTION, {})
        # we prefer the action name but will use action text, if there is no name
        action = prev_action.get(ACTION_NAME, None)
        if not action:
            action = prev_action.get(ACTION_TEXT, None)
        if not action:
            raise RuntimeError("There must be an action we can extract....")
        if self.remove_last_turn:
            turns = turns[:-1]
        else:
            if not inplace_allowed:
                turns[-1] = copy.deepcopy(turns[-1])
            turns[-1].state.pop(PREVIOUS_ACTION, {})
        return turns, action

    def from_domain(self, domain: Domain,) -> List[Text]:
        return domain.action_names_or_texts
