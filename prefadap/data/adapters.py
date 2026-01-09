"""Adapters for aligning external datasets to the internal preference schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Union

from datasets import Dataset, DatasetDict


@dataclass(frozen=True)
class ColumnMapping:
    """Map canonical fields to their column names in a source dataset.

    Each attribute stores the **original** column name that corresponds to the
    canonical preference fields ``prompt``, ``chosen`` and ``rejected``.  This
    orientation lets callers specify how their dataset labels these concepts so
    the adapter can rename them to the internal schema.
    """

    prompt: str
    chosen: str
    rejected: str

    @classmethod
    def from_dict(cls, data: Mapping[str, str]) -> "ColumnMapping":
        """Create a :class:`ColumnMapping` from a plain mapping."""

        return cls(
            prompt=data["prompt"],
            chosen=data["chosen"],
            rejected=data["rejected"],
        )


def _rename(dataset: Dataset, mapping: ColumnMapping) -> Dataset:
    rename = {
        mapping.prompt: "prompt",
        mapping.chosen: "chosen",
        mapping.rejected: "rejected",
    }
    return dataset.rename_columns(
        {src: dst for src, dst in rename.items() if src in dataset.column_names}
    )


def standardize_preference_dataset(
    ds: Union[Dataset, DatasetDict], mapping: ColumnMapping
) -> Union[Dataset, DatasetDict]:
    """Rename columns of ``ds`` to the canonical preference keys.

    Parameters
    ----------
    ds:
        A :class:`datasets.Dataset` or :class:`datasets.DatasetDict` with
        arbitrary column names.
    mapping:
        A :class:`ColumnMapping` describing the dataset's column names.

    Returns
    -------
    Union[Dataset, DatasetDict]
        The dataset with columns renamed to ``prompt``, ``chosen`` and
        ``rejected``.
    """

    if isinstance(ds, DatasetDict):
        return DatasetDict({split: _rename(d, mapping) for split, d in ds.items()})
    return _rename(ds, mapping)


__all__ = ["ColumnMapping", "standardize_preference_dataset"]

