#!/usr/bin/env bash
rm -rf build/ dist/
python3 setup.py bdist_wheel
twine check dist/*
twine upload dist/*