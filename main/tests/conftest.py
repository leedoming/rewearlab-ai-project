import os
import sys

# Make `import evaluation` and `import retrieval` work regardless of the
# directory pytest is invoked from. Both packages live under main/,
# alongside this tests/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
