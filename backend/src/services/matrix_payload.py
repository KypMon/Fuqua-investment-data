from dataclasses import dataclass
from typing import List, Dict
import pandas as pd

@dataclass
class MatrixPayload:
    columns: List[str]
    records: List[Dict[str, object]]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "MatrixPayload":
        return cls(columns=list(df.columns), records=df.to_dict(orient="records"))

    def to_dataframe(self) -> pd.DataFrame:
        if not self.columns:
            raise ValueError("No columns provided in payload")
        if not isinstance(self.records, list):
            raise ValueError("Matrix payload expects a list of records")
        return pd.DataFrame(self.records, columns=self.columns)