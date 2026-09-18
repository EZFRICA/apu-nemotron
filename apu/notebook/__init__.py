"""The student notebook: what the student chose to keep, and their revision sheets.

During a conversation the student saves what matters to them, from a button under an answer
or by asking the tutor ("save that"). Each save keeps one of three things, the student's
choice: the full answer, its key points (condensed by Nemotron), or an excerpt they pick.
A revision sheet is then made from the notebook: the entries the student selects, as
written, or a summary of them written by Nemotron.

The tutor never reads the notebook. Nothing in apu.runtime.agent loads entries into a
prompt, and the store lives in its own SQLite file, apart from the DLL and the L3 tables
the BMJ search reads. The only thing the tutor does with it is write, when asked.
"""
