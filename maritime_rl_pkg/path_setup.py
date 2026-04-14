import os
import sys

def ensure_paths():
    here = os.path.dirname(os.path.abspath(__file__))

    # Walk up until we find the folder that contains third_party/CORALL
    repo_root = here
    while True:
        candidate = os.path.join(repo_root, "third_party", "CORALL", "src")
        if os.path.isdir(candidate):
            break
        parent = os.path.dirname(repo_root)
        if parent == repo_root:
            raise RuntimeError("Could not find third_party/CORALL/src by walking up from: " + here)
        repo_root = parent

    corall_root = os.path.join(repo_root, "third_party", "CORALL")
    corall_src  = os.path.join(corall_root, "src")

    # Put CORALL first (so CORALL's `src.*` wins if needed)
    for p in [corall_root, corall_src, repo_root]:
        if p not in sys.path:
            sys.path.insert(0, p)

    if os.environ.get("PRINT_PATHS", "1") == "1":
        print("repo_root =", repo_root)
        print("corall_root =", corall_root)
        print("corall_src  =", corall_src)
        print("sys.path[:6] =", sys.path[:6])