"""Pluggable range API. Stereo disparity can be implemented behind this interface."""
from abc import ABC, abstractmethod
from typing import Optional
from argos.models import RangeEstimate


class RangeProvider(ABC):
    @abstractmethod
    def estimate(self, left_observation, right_observation) -> Optional[RangeEstimate]:
        """Return range in meters, or None when evidence is insufficient."""


class UnavailableRangeProvider(RangeProvider):
    def estimate(self, left_observation, right_observation):
        return None
