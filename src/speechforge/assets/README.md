`selftest-voice.wav`: 12.6 s of fully synthetic speech (Kokoro-82M TTS, voice
`am_michael`, Apache-2.0 weights; the DisPatch voice add-on's stock clip). It is
not a recording of a real person, so building Chatterbox conditionals from it on
the rig does not touch decision D3 (family reference clips stay on the box).

Used by the post-load self-test and the benchmark: it is the golden STT clip
(expected text in `selftest.py`) and the reference of the synthetic test voice.
