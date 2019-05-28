#!/bin/bash
# pre-commit.sh
# Runs tests and returns non-zero exit code if they failed

# Stash any changes to make sure they don't interfere with tests
STASH_NAME="pre-commit-$(date +%s)"
git stash save -q --keep-index $STASH_NAME

# Test prospective commit
./run_tests.sh
TEST_RESULT=$?

# Run pylint on all .py files and check no errors
find . -iname "*.py" | xargs pylint --rcfile=.pylintrc
PYLINT_RESULT=$?

# Pop the stashed changes
STASHES=$(git stash list)
if [[ $STASHES == "$STASH_NAME" ]]; then
  git stash pop -q
fi

# If either test failed, exit with code 1 so commit is halted
[ $TEST_RESULT -ne 0 ] || [ $PYLINT_RESULT -ne 0 ] && exit 1
exit 0
