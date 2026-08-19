---
title: CivicStruct grievance structurer
emoji: 🏛️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.0.1
python_version: 3.12
app_file: app.py
startup_duration_timeout: 30m
short_description: Structure a fictional civic complaint as validated JSON.
models:
  - HuggingFaceTB/SmolLM3-3B
preload_from_hub:
  - HuggingFaceTB/SmolLM3-3B
---

# CivicStruct

This demo turns one fictional public-service complaint into a structured JSON
record and a neutral summary. It uses the frozen CivicStruct QLoRA adapter and
keeps schema failures visible.

Use fictional complaints only. The model can still make extraction mistakes,
so the output needs human review before anyone acts on it.
