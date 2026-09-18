import os
import sys

# Make `import evaluation` (and, once Milestone 1 is merged, `import retrieval`)
# work regardless of the directory pytest is invoked from - both packages
# live under main/, alongside this tests/ dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
