test:
    # Lint python files
    find . -type f -name "*.py" -exec pylint -j 0 --exit-zero {} \;
