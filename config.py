from pathlib import Path

RESULT_DIR = Path('results/raw')
FIGURE_DIR = Path('figures')
TABLE_DIR = Path('tables')

TABLE_DIR.mkdir(exist_ok=True, parents=True)