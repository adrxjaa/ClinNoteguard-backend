import sys
from pathlib import Path

# Ensure clinnote_backend root is in sys.path
backend_root = Path(__file__).parent.parent.resolve()
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))
