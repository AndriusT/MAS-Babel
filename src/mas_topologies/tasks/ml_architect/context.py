"""Context preparation and response parsing for the ML architecture design task."""

import json
import re
from typing import Any, Dict

from .models import DesignInput, DesignOutput


def _prepare_feedback_section(feedback: Dict[str, Any]) -> str:
    """Format previous training feedback for the designer context."""
    section = "\n## PREVIOUS ITERATION FEEDBACK\n\n"

    section += f"""### Previous Performance
- **Final Validation Accuracy**: {feedback.get('final_val_accuracy', 0):.2%}
- **Best Validation Accuracy**: {feedback.get('best_val_accuracy', 0):.2%}
- **Training Accuracy**: {feedback.get('final_train_accuracy', 0):.2%}
- **Overfitting Gap**: {feedback.get('overfitting_gap', 0):.2%}
- **Convergence Speed**: {feedback.get('convergence_speed', 'unknown')}
- **Performance Rating**: {feedback.get('performance_rating', 'unknown').upper()}
"""

    if feedback.get("issues"):
        section += "\n### Issues Detected\n"
        for issue in feedback["issues"]:
            section += f"- {issue}\n"
    else:
        section += "\n### Issues Detected\nNone \u2014 model performed well\n"

    if feedback.get("recommendations"):
        section += "\n### Recommendations for Next Iteration\n"
        for rec in feedback["recommendations"]:
            section += f"- {rec}\n"

    if feedback.get("next_steps"):
        section += "\n### Suggested Next Steps\n"
        for step in feedback["next_steps"]:
            section += f"- {step}\n"

    section += """
## YOUR TASK FOR THIS ITERATION

Based on the feedback above, propose an IMPROVED architecture that addresses the issues
and incorporates the recommendations.

Focus on:
1. What went wrong in the previous iteration?
2. How can the architecture be modified to fix these issues?
3. What specific changes would likely improve performance?
4. Balance between model complexity and overfitting
"""
    return section


def prepare_context(design_input: DesignInput) -> str:
    """Build the full context string passed to the LLM."""
    context = f"""# ML Architecture Design Task

## Dataset Information
- **Name**: {design_input.dataset_name}
- **Description**: {design_input.dataset_description}
- **Number of Classes**: {design_input.num_classes}
- **Input Shape**: {design_input.input_shape} (HWC format)
- **Dataset Size**: {design_input.dataset_size} samples
"""

    if design_input.iteration_number > 1 and design_input.previous_feedback:
        context += _prepare_feedback_section(design_input.previous_feedback)

    if design_input.data_format_info:
        info = design_input.data_format_info
        context += f"""
## Data Format Details
- **Format**: {info.data_format}
- **Shape**: {info.shape} (actual training data format)
- **Value Range**: [{info.value_range[0]:.3f}, {info.value_range[1]:.3f}]
- **Number of Channels**: {info.shape[1] if len(info.shape) > 1 else 'unknown'}
- **Spatial Dimensions**: {info.shape[2] if len(info.shape) > 2 else '?'} x {info.shape[3] if len(info.shape) > 3 else '?'}
"""

    if design_input.framework:
        context += f"""
## Framework & Environment
- **Framework**: {design_input.framework.upper()}
- **Version**: {design_input.framework_version}
"""

    if design_input.initial_architecture:
        context += f"\n## Initial Architecture Idea\n{design_input.initial_architecture}\n"

    if design_input.constraints:
        context += "\n## Constraints\n"
        for key, value in design_input.constraints.items():
            context += f"- {key}: {value}\n"

    if design_input.papers_content:
        context += "\n## Relevant Research Papers\n"
        for i, paper in enumerate(design_input.papers_content, 1):
            truncated = paper[:5000] if len(paper) > 5000 else paper
            context += f"\n### Paper {i}\n{truncated}\n"

    if design_input.user_notes:
        context += f"\n## Additional Notes\n{design_input.user_notes}\n"

    context += f"""
## Task
Analyze the above dataset properties and propose an optimal neural network architecture.

IMPORTANT:
- input_shape={design_input.input_shape} (HWC format) will be passed to create_model()
- The data_format_info shows the actual training data format for reference
- Design your architecture to work with the HWC input_shape format
- Provide implementation-ready code and specifications

Your response MUST include:
1. Analysis of dataset properties and design justifications
2. Complete Python code in a code block (see REQUIRED OUTPUT FORMAT above)
3. JSON hyperparameters block with training specifications
"""
    return context


def parse_response(response_text: str, design_input: DesignInput) -> DesignOutput:
    """Parse raw LLM response text into a structured DesignOutput."""
    hyperparams: Dict[str, Any] = {
        "framework": design_input.framework,
        "input_shape": design_input.input_shape,
        "num_classes": design_input.num_classes,
    }

    json_pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(json_pattern, response_text, re.DOTALL)

    if matches:
        try:
            json_str = re.sub(r"\s*//[^\n]*", "", matches[-1])
            data = json.loads(json_str)
            if "hyperparameters" in data:
                hp = data["hyperparameters"]
                hyperparams.update(
                    {
                        "optimizer": hp.get("optimizer", "adam"),
                        "learning_rate": hp.get("learning_rate", 0.001),
                        "weight_decay": hp.get("weight_decay", 0.0),
                        "batch_size": hp.get("batch_size", 64),
                        "epochs": hp.get("epochs", 20),
                        "learning_rate_schedule": hp.get("learning_rate_schedule", "none"),
                        "warmup_epochs": hp.get("warmup_epochs", 0),
                        "dropout_rate": hp.get("dropout_rate", 0.0),
                    }
                )
        except json.JSONDecodeError:
            print("Warning: could not parse hyperparameters JSON from Designer")
    else:
        print("Warning: no hyperparameters JSON block found in Designer response")

    return DesignOutput(
        proposed_architecture=response_text,
        rationale="See full response for detailed analysis",
        hyperparameters=hyperparams,
        preprocessing_steps=[
            "Normalization based on value range",
            "Train/validation/test split",
        ],
        training_strategy="See full response for detailed training strategy",
        references=[],
    )


__all__ = ["prepare_context", "parse_response", "_prepare_feedback_section"]
