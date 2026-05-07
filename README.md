# IPIBench

Welcome to IPIBench! This repository contains the evaluation and inference scripts for IPIBench. 

## Structure

The project is organized into the following task hierarchy:

- **IRP** (Instruction Representation & Processing)
  - `RAP`: Reactive-after-Proactive Interaction
  - `RTP`: Reactive-to-Proactive Interaction
  - `RUP`: Reactive-under-Proactive Interaction
- **PM** (Property Matching)
  - `PT`: Proactive Timing
  - `PU`: Proactive Understanding
  - `RP`: Repeated Proactiveness
- **PTM** (Prompt & Task Modification)
  - `MTM`: Multi-task Management
  - `TC`:  Task Cancellation
  - `TM`:  Task Modification

## Directories

- `Src/`: Contains the inference scripts for all models. Each model has its own directory with a `run.sh` script to execute the inference process across all tasks.
- `Evaluate/`: Contains the evaluation scripts to calculate scores based on the inference results. Use `run_eval.sh` to run the evaluation across all models and tasks.

## Usage

1. **Inference**: Navigate to the specific model directory inside `Src/` (e.g., `Src/Gemini_3/`) and run the corresponding script:
   ```bash
   bash run.sh
   ```

2. **Evaluation**: After generating the results, navigate to the `Evaluate/` directory and run the evaluation script to calculate the metrics:
   ```bash
   bash run_eval.sh
   ```

*Note: The code related to the Agent framework is still being organized and will be open-sourced in a future update.*
