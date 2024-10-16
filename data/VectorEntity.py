from dataclasses import dataclass
from typing import List


@dataclass
class VectorEntity:
    id: int
    skill_ratio_first_scrape: List[float]
    minigame_ratio_first_scrape: List[float]
    skill_gain_ratio: List[float]
    minigame_gain_ratio: List[float]

    def __repr__(self):
        return f'{self.id}'

    def __eq__(self, other):
        return self.id != other.id