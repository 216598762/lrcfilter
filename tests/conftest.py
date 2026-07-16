"""Root conftest for lrcfilter tests.

Mocks faster_whisper and numpy at the session level to avoid import issues
in test environments where these libraries may have conflicts.
"""

import sys
from unittest.mock import MagicMock

# Mock faster_whisper before any test module imports it
# This prevents the numpy "cannot load module more than once" error
if "faster_whisper" not in sys.modules:
    mock_faster_whisper = MagicMock()
    mock_faster_whisper.WhisperModel = MagicMock
    sys.modules["faster_whisper"] = mock_faster_whisper
