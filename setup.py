"""
Copyright (c) 2022-2023 yewentao
Licensed under the MIT License.
"""

import os
import subprocess
import sys
from pathlib import Path
import logging

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = "0.4.0"

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        debug = self.debug or int(os.environ.get("DEBUG", 0))
        use_cuda = os.environ.get("CUDA", "OFF")
        cfg = "Debug" if debug else "Release"

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            f"-DCUDA={use_cuda}",
        ]
        build_args = []

        if self.compiler.compiler_type == "msvc":
            cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]
            cmake_args += [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
            ]
            build_args += ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        logging.info('######sub process running info######')
        p1 = subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        logging.info(f'Executing CMake command: `{" ".join(p1.args)}`')
        logging.info(p1.stdout.decode('utf-8', 'ignore'))
        p1.check_returncode()
        p2 = subprocess.run(
            ["cmake", "--build", "."] + build_args, cwd=build_temp,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        logging.info(f'Executing CMake command: `{" ".join(p2.args)}`')
        logging.info(p2.stdout.decode('utf-8', 'ignore'))
        p2.check_returncode()
        logging.info('######end of sub process running info######')


setup(
    name="microtorch",
    version=__version__,
    author="yewentao",
    author_email="zhyanwentao@outlook.com",
    description="MicroTorch: A simplest pytorch implementation for learning",
    ext_modules=[CMakeExtension("_microtorch")],
    cmdclass={"build_ext": CMakeBuild},
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.8",
    packages=['microtorch'],
)
