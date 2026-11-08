import difflib
import pathlib
import subprocess
import sys

repo = pathlib.Path(sys.argv[1])
working_source = pathlib.Path(sys.argv[2]) / "av1/encoder/intra_mode_search.c"
original = subprocess.check_output([
    "git", "-C", str(repo / "research_pass35/sources/libaom"), "show",
    "10aece4157eb79315da205f39e19bf6ab3ee30d0:av1/encoder/intra_mode_search.c",
], text=True)
modified = working_source.read_text(encoding="utf-8")
diff = difflib.unified_diff(
    original.splitlines(keepends=True), modified.splitlines(keepends=True),
    fromfile="a/av1/encoder/intra_mode_search.c",
    tofile="b/av1/encoder/intra_mode_search.c",
)
pathlib.Path(sys.argv[3]).write_text("".join(diff), encoding="utf-8")
