#!/bin/bash
# pre-commit.sh
# Runs tests and returns non-zero exit code if they failed

# Test prospective commit
./run_tests.sh
TEST_RESULT=$?

# Run pylint on all .py files and check no errors
find . -iname "*.py" | xargs pylint --rcfile=.pylintrc
PYLINT_RESULT=$?

# If either test failed, exit with code 1 so commit is halted
[ $TEST_RESULT -ne 0 ] || [ $PYLINT_RESULT -ne 0 ] && exit 1
exit 0
