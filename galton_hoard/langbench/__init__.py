"""The language bench: the same tool-using tasks in Spanish and English, with the metrics that tell a language gap from a model gap.

Each task has the same id, tools, argument names and enum values in both languages. What is translated is decided by type (see ``canon``): ids,
names and enums are preserved, instructions and tool descriptions are written afresh, and the entities shared by prompt, data and checks come from
one canonical map. ``tasks.localize`` gives a task in the shape the harness bench runs; ``judge`` measures the language of a reply and of a tool
argument; ``shop`` and ``family`` are the two data sets, searched with BM25.
"""

from .canon import BASE_LANG, LANGS, TYPOLOGY
from .judge import arg_matches_data, consistency
from .tasks import I18N_TASK_IDS, KINDS, SPECS_BY_ID, localize, public, select

__all__ = ["BASE_LANG", "I18N_TASK_IDS", "KINDS", "LANGS", "SPECS_BY_ID", "TYPOLOGY", "arg_matches_data", "consistency", "localize", "public", "select"]
