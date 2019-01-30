import pendulum as pend
import csv


def min_max_normalization(old_value: float,
                          old_range: dict,
                          new_range: dict
                          ) -> float:
    new_value = ((old_value - old_range["min"])
                 / (old_range["max"] - old_range["min"])) \
                * (new_range["max"] - new_range["min"]) \
                + new_range["min"]
    return new_value


