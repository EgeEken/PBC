from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
out=root.parent/'artifacts/mse.dll'
subprocess.run(['gcc','-O3','-std=c11','-Wall','-Wextra','-Werror','-shared','-static-libgcc',str(root/'mse.c'),'-o',str(out)],check=True)
print(out)
