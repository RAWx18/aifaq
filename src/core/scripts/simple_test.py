#!/usr/bin/env python3
"""
Simple test script for the multi-agent system.
"""

import sys
import os

# Simple test script
def main():
    print("Simple test script running...")
    print("Path:", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Test completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
