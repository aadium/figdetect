# Rhetorical Figure Detection

This repository contains code to train a rhetorical figure detector.

All data is extracted from here: [rhetoricon.uwaterloo.ca/admin/export](https://rhetoricon.uwaterloo.ca/admin/export)

Store the 3 json files in the `input` directory.

Run `pip freeze > requirements.txt`.

Then, run `extract.py`, `english.py`, and `phonetize.py` in order to generate the CSV file usable for training.

Then, run all cells in `train.ipynb`.