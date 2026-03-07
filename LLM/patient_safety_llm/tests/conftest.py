import sys
import os

# Ensure the package root (patient_safety_llm) is on sys.path so tests can import `src`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
