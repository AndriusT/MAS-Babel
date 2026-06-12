"""Pydantic models for the ML architecture design task."""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel


class DataFormatInfo(BaseModel):
    """Information about the dataset's raw tensor format."""

    shape: Tuple[int, ...]
    dtype: str
    data_format: str          # "HWC" or "BCHW"
    value_range: Tuple[float, float]
    num_samples: int
    num_classes: int


class DesignInput(BaseModel):
    """All information the designer needs to propose an architecture."""

    dataset_name: str
    dataset_description: str
    num_classes: int
    input_shape: Tuple[int, ...]
    dataset_size: int
    initial_architecture: Optional[str] = None
    papers_content: Optional[List[str]] = None
    user_notes: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    framework: Optional[str] = None
    framework_version: Optional[str] = None
    data_format_info: Optional[DataFormatInfo] = None
    iteration_number: int = 1
    previous_feedback: Optional[Dict[str, Any]] = None
    previous_training_results: Optional[List[Dict[str, Any]]] = None


class DesignOutput(BaseModel):
    """Structured output from the designer."""

    proposed_architecture: str
    rationale: str
    hyperparameters: Dict[str, Any]
    preprocessing_steps: List[str]
    training_strategy: str
    references: List[str]


__all__ = ["DataFormatInfo", "DesignInput", "DesignOutput"]
