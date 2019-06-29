#!/bin/bash
# pre-commit.sh
# Runs tests and returns non-zero exit code if they failed

# Run pylint on all .py files and check no errors
find . -iname "scripts/*.py" | xargs pylint --rcfile=.pylintrc
PYLINT_RESULT=$?

# Test prospective commit
$(coverage run -m unittest discover -k test_short)
TEST_RESULT=$?

COVERAGE_RESULT=0

# If either test failed, exit with code 1 so commit is halted
[ $TEST_RESULT -ne 0 ] || [ $COVERAGE_RESULT -ne 0 ] || [ $PYLINT_RESULT -ne 0 ] && exit 1
exit 0
