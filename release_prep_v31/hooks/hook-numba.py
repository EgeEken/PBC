from PyInstaller.utils.hooks import collect_submodules
hiddenimports=collect_submodules('numba',filter=lambda n: not any(n==p or n.startswith(p+'.') for p in ('numba.tests','numba.cuda','numba.pycc')))
excludedimports=['scipy','IPython','setuptools','pkg_resources']
