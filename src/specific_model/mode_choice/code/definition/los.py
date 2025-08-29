from typing import Any, Hashable
from dataclasses import dataclass
import re

__all__ = ["Los"]


@dataclass
class Los:
    o_zone: int
    d_zone: int
    availability: dict[int, bool]  # key: mode, value: availability
    attribute_names: list[str]  # list of attribute names
    attributes: dict[int, list[float]]  # key: mode, value: list of attribute values

    @staticmethod
    def from_dict(data: dict[Hashable, Any]) -> "Los":
        o_zone = int(data["OZone"])
        d_zone = int(data["DZone"])

        # keyのうち{mode番号}Availableの形式のものを抽出
        pattern = r"^(\d+)Available$"
        availability = {
            int(m.group(1)): bool(data[k])
            for k in data.keys()
            if (m := re.match(pattern, str(k)))
        }
        # keyのうち{mode番号}{属性名}の形式になっているものを抽出
        pattern_attr = r"^(\d+)(.*)$"
        attribute_names = set([m.group(2) for k in data.keys() if (m := re.match(pattern_attr, str(k))) and not re.match(pattern, str(k))])
        ## sort
        attribute_names = sorted(list(attribute_names))
        attributes = {int(mode): [0.0] * len(attribute_names) for mode in availability.keys()}
        for mode in attributes.keys():
            for i, att_name in enumerate(attribute_names):
                if (value := data.get(f"{mode}{att_name}", None)) is not None:
                    attributes[int(mode)][i] = float(value)

        return Los(
            o_zone=o_zone,
            d_zone=d_zone,
            availability=availability,
            attributes=attributes,
            attribute_names=attribute_names
        )