@echo off
call .venv\Scripts\activate
python -m src.predict --input-dir data\test --output outputs\test_predictions.csv
python -m src.evaluate --truth data\test.csv --predictions outputs\test_predictions.csv
pause
