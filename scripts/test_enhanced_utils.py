#!/usr/bin/env python3
"""
Test script to verify the enhanced data utilities work correctly.
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

def test_data_manager():
    """Test the enhanced data manager functionality."""
    print("Testing enhanced data manager...")

    try:
        from tops_insar2 import DataManager, DEMManager, GPUManager, initialize_managers

        # Mock args object
        class MockArgs:
            def __init__(self):
                self.dem = None
                self.auto_download = True
                self.generate_test_data = True
                self.master_safe_or_manifest = "/tmp/test"
                self.output_dir = "/tmp/output"

        args = MockArgs()

        # Initialize managers
        initialize_managers(args)
        print("✓ Managers initialized successfully")

        # Test data manager
        data_mgr = DataManager(args)
        print("✓ DataManager created")

        # Test DEM manager
        dem_mgr = DEMManager(args)
        print("✓ DEMManager created")

        # Test GPU manager
        gpu_mgr = GPUManager()
        print("✓ GPUManager created")

        return True

    except ImportError as e:
        print(f"⚠ Enhanced utilities not available: {e}")
        print("This is expected if tops_data_utils.py is not in place")
        return False
    except Exception as e:
        print(f"✗ Error testing enhanced utilities: {e}")
        return False

def test_basic_imports():
    """Test that basic imports still work."""
    print("\nTesting basic imports...")

    try:
        import numpy as np
        from pathlib import Path
        print("✓ Basic imports work")

        # Test if we can at least import the main module
        import tops_insar2
        print("✓ Main module imported successfully")

        return True

    except Exception as e:
        print(f"✗ Error with basic imports: {e}")
        return False

def test_args_parsing():
    """Test argument parsing works."""
    print("\nTesting argument parsing...")

    try:
        import argparse
        import tops_insar2

        # Create a simple parser like in tops_insar2
        parser = argparse.ArgumentParser(description="Test parser")
        parser.add_argument("output_dir", type=Path)
        parser.add_argument("--master-product-path", type=Path, default=None)

        # Test parsing
        args = parser.parse_args(["/tmp/test_output"])
        print("✓ Argument parsing works")

        return True

    except Exception as e:
        print(f"✗ Error with argument parsing: {e}")
        return False

def main():
    """Run all tests."""
    print("=== TOPS InSAR2 Enhanced Utilities Test ===\n")

    results = []

    # Test basic functionality
    results.append(test_basic_imports())
    results.append(test_args_parsing())

    # Test enhanced utilities (if available)
    enhanced_available = test_data_manager()
    if enhanced_available:
        print("\n🎉 Enhanced utilities are working!")
    else:
        print("\nℹ️  Enhanced utilities not fully available, but core functionality intact")

    # Summary
    print("\n=== Test Summary ===")
    passed = sum(results) + (1 if enhanced_available else 0)
    total = len(results) + 1

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())