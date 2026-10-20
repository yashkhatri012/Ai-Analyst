
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class IngestedDataset:
    dataframe: pd.DataFrame
    metadata: dict[str, Any]
