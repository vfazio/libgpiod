# SPDX-License-Identifier: GPL-2.0-or-later
# SPDX-FileCopyrightText: 2022 Bartosz Golaszewski <brgl@bgdev.pl>

from os import getenv

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as orig_build_ext

LINK_SYSTEM_LIBGPIOD = getenv("LINK_SYSTEM_LIBGPIOD") == "1"
LIBGPIOD_VERSION = getenv("LIBGPIOD_VERSION")


class build_ext(orig_build_ext):
    """
    Wrap build_ext to amend the module sources and settings to build
    the bindings and gpiod into a combined module when a version is
    specified and LINK_SYSTEM_LIBGPIOD=1 is not present in env.

    run is wrapped with @fetch_tarball in order to fetch the sources
    needed to build binary wheels when LIBGPIOD_VERSION is specified, eg:

    LIBGPIOD_VERSION="2.0.2" python3 -m build .
    """

    def run(self):
        # Try to get the gpiod version from the .txt file included in sdist
        try:
            libgpiod_version = open("libgpiod-version.txt", "r").read()
        except OSError:
            libgpiod_version = LIBGPIOD_VERSION

        if libgpiod_version and not LINK_SYSTEM_LIBGPIOD:
            # When building the extension from an sdist with a vendored
            # amend gpiod._ext sources and settings accordingly.
            gpiod_ext = self.ext_map["gpiod._ext"]
            gpiod_ext.sources += [
                "lib/chip.c",
                "lib/chip-info.c",
                "lib/edge-event.c",
                "lib/info-event.c",
                "lib/internal.c",
                "lib/line-config.c",
                "lib/line-info.c",
                "lib/line-request.c",
                "lib/line-settings.c",
                "lib/misc.c",
                "lib/request-config.c",
            ]
            gpiod_ext.libraries = []
            gpiod_ext.include_dirs = ["include", "lib", "gpiod/ext"]
            gpiod_ext.extra_compile_args.append(
                f'-DGPIOD_VERSION_STR="{libgpiod_version}"',
            )

        super().run()

gpiod_ext = Extension(
    "gpiod._ext",
    sources=[
        "gpiod/ext/chip.c",
        "gpiod/ext/common.c",
        "gpiod/ext/line-config.c",
        "gpiod/ext/line-settings.c",
        "gpiod/ext/module.c",
        "gpiod/ext/request.c",
    ],
    define_macros=[("_GNU_SOURCE", "1")],
    libraries=["gpiod"],
    extra_compile_args=["-Wall", "-Wextra"],
)

setup(
    ext_modules=[gpiod_ext],
    cmdclass={"build_ext": build_ext},
)
