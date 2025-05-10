# AIFAQ Scripts

This directory contains utility scripts and tests for the AIFAQ project.

## Available Scripts

### `test_multi_agent.py`

Tests the multi-agent RAG system by processing sample queries through the pipeline. This script initializes the models, creates the multi-agent system, and runs test queries to verify functionality.

**Usage:**
```bash
python test_multi_agent.py
```

### `test_agent_coordination.py`

Tests the coordination between agents in the multi-agent RAG system using mocks instead of loading the full language model. This is useful for quick validation of agent interactions without the overhead of downloading and loading large models.

**Usage:**
```bash
python test_agent_coordination.py
```

### `format.sh`

A shell script to format Python code in the project using Black.

**Usage:**
```bash
./format.sh
```

## Adding New Scripts

When adding new scripts to this directory, please follow these guidelines:

1. Use descriptive filenames that indicate the script's purpose
2. Include proper documentation within the script
3. Add the script to this README file with a brief description and usage instructions
4. Make sure the script is executable if it's a shell script (`chmod +x script_name.sh`)
