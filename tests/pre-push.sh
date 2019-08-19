#!/bin/bash
# pre-commit.sh
# Runs tests and returns non-zero exit code if they failed

# Run pylint on all .py files and check no errors
find ./scripts -name "*.py" | xargs pylint --rcfile=.pylintrc
PYLINT_RESULT=$?

# Test prospective commit
$(coverage run --source=scripts/helper -m unittest discover -s tests)
TEST_RESULT=$?

coverage report --fail-under=80
COVERAGE_RESULT=$?

KCN_CURRENT_DIR=$(pwd) snakemake --dag | dot -Tpdf > dag.pdf
SNAKEMAKE_RESULT=$?

# If any test failed, exit with code 1 so commit is halted
[ $SNAKEMAKE_RESULT -ne 0 ] || [ $TEST_RESULT -ne 0 ] || [ $COVERAGE_RESULT -ne 0 ] || [ $PYLINT_RESULT -ne 0 ] && exit 1
exit 0
