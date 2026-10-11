"""The family listening-test kit (SPEC §11.1 "Listening", §12 phase 2).

Five steps, each a script run from the repository root with
``uv run python -m tools.listening.<step> ...``:

1. ``render_set``  -- render ``lines.txt`` through a running SpeechForge, one engine at a time
2. ``make_set``    -- blind, shuffle and loudness-match the renders; write the manifest and the key
3. ``serve``       -- the rating page the family opens on the home network
4. ``score``       -- MOS, preference win rate, confidence intervals, ``report.md``
5. ``wer`` / ``similarity`` -- the objective checks from SPEC §11.3 (optional)

No audio, ratings or keys are ever committed; see ``docs/LISTENING-TEST.md``.
"""
