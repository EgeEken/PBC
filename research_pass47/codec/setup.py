from setuptools import Distribution, setup

class WindowsDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(distclass=WindowsDistribution)
