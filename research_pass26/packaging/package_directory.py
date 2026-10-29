from pathlib import Path
import shutil
root=Path(__file__).resolve().parent
folder=root.parent/'artifacts/onedir/pbc'
shutil.copy2(root/'USAGE.txt',folder/'USAGE.txt')
archive=shutil.make_archive(str(root.parent/'artifacts/PBC32Research-Windows-x64'),'zip',folder.parent,folder.name)
print(archive)
