#!/usr/bin/env python3
"""
Backward-Compatibility Wrapper for src/extract_workflow.py

This module redirects to src/extract_workflow.py, which serves as the unified single-file 
and multi-file schema extraction engine.
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import extract_workflow


async def async_main():
    await extract_workflow.async_main()


def main():
    extract_workflow.main()


if __name__ == "__main__":
    main()
