import sys
from pathlib import Path
import subprocess
PASS=Path(__file__).resolve().parents[1]
source=PASS/'artifacts/candidate/pbc3'
subprocess.run(['gcc','-O3','-std=c11','-Wall','-Wextra','-Werror','-shared','-static-libgcc',str(source/'mse.c'),'-o',str(source/'mse.dll')],check=True)
print(source/'mse.dll')
