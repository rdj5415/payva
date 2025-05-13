#!/bin/bash
# Script to run tests with the correct environment variables

# Exit on error
set -e

# Enable verbose mode
set -x

# Load test environment variables
set -a
source .env.test

# Ensure SQLite is accepted for testing
export DATABASE_URL=sqlite+aiosqlite:///:memory:
export DATABASE_TEST_URL=sqlite+aiosqlite:///:memory:
export PYTHONPATH=$(pwd):$PYTHONPATH
set +a

# Print environment variables for debugging
echo "Database URL: $DATABASE_URL"
echo "Database Test URL: $DATABASE_TEST_URL"
echo "PYTHONPATH: $PYTHONPATH"

# Create a simple pytest file with a dummy test to verify setup
cat > test_setup.py << EOF
import pytest

def test_setup():
    """Verify that the test environment is set up correctly."""
    assert True
EOF

# Run the simple test to verify pytest setup
pytest -xvs test_setup.py

# Remove the temporary test file
rm test_setup.py

# Run the rules engine tests
pytest -xvs tests/rules_engine/test_rules_engine.py "$@" 