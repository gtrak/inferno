"""`inspect.py` — print the module tree and parameter table of a Qwen4Exp config (I-04).

    python inspect.py [config.json]

The file name is mandated by issue I-04, and it shadows the stdlib `inspect` module that
`torch` imports. Any python process started from this directory would therefore break
(`import inspect` would load this file instead of the stdlib module). To keep that from
happening, this module installs the *real* stdlib `inspect` into `sys.modules` before doing
anything else, and only runs the tool when executed as a script; the tool itself lives in
`module_tree.py`.
"""

import importlib.util
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HIDDEN = [entry for entry in sys.path if os.path.abspath(entry or os.getcwd()) == _SCRIPT_DIR]
sys.path[:] = [entry for entry in sys.path if os.path.abspath(entry or os.getcwd()) != _SCRIPT_DIR]
_spec = importlib.util.spec_from_file_location("inspect", os.path.join(os.path.dirname(os.__file__), "inspect.py"))
_stdlib_inspect = importlib.util.module_from_spec(_spec)
sys.modules["inspect"] = _stdlib_inspect
_spec.loader.exec_module(_stdlib_inspect)
sys.path[0:0] = _HIDDEN


if __name__ == "__main__":
    import module_tree

    sys.exit(module_tree.main(sys.argv))
