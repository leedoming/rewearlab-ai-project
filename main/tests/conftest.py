import os
import sys

# Make `import retrieval` work regardless of the directory pytest is
# invoked from (retrieval/ lives under main/, alongside this tests/ dir).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
