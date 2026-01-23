from os import getenv, path, unlink
from shutil import copy, copytree, rmtree

from setuptools import build_meta as _orig
from setuptools.build_meta import *
from setuptools.command.sdist import log
from setuptools.errors import BaseError

LINK_SYSTEM_LIBGPIOD = getenv("LINK_SYSTEM_LIBGPIOD") == "1"
LIBGPIOD_MINIMUM_VERSION = "2.1"
LIBGPIOD_VERSION = getenv("LIBGPIOD_VERSION")
SRC_BASE_URL = "https://mirrors.edge.kernel.org/pub/software/libs/libgpiod/"
TAR_FILENAME = "libgpiod-{version}.tar.gz"
ASC_FILENAME = "sha256sums.asc"
SHA256_CHUNK_SIZE = 2048


def sha256(filename):
    """
    Return a sha256sum for a specific filename, loading the file in chunks
    to avoid potentially excessive memory use.
    """
    from hashlib import sha256

    sha256sum = sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(SHA256_CHUNK_SIZE), b""):
            sha256sum.update(chunk)

    return sha256sum.hexdigest()


def find_sha256sum(asc_file, tar_filename):
    """
    Search through a local copy of sha256sums.asc for a specific filename
    and return the associated sha256 sum.
    """
    with open(asc_file) as f:
        for line in f:
            line = line.strip().split("  ")
            if len(line) == 2 and line[1] == tar_filename:
                return line[0]

    raise BaseError(f"no signature found for {tar_filename}")


def fetch_tarball(command, *args):
    """
    Verify the requested LIBGPIOD_VERSION tarball exists in sha256sums.asc,
    fetch it from https://mirrors.edge.kernel.org/pub/software/libs/libgpiod/
    and verify its sha256sum.

    If the check passes, extract the tarball and copy the lib and include
    dirs into our source tree.
    """

    # If no LIBGPIOD_VERSION is specified in env, just run the command
    if LIBGPIOD_VERSION is None:
        return command

    # If LIBGPIOD_VERSION is specified, apply the tarball wrapper
    def wrapper(self, *args):
        # Just-in-time import of tarfile and urllib.request so these are
        # not required for Yocto to build a vendored or linked package
        import sys
        import tarfile
        from tempfile import TemporaryDirectory
        from urllib.request import urlretrieve

        from packaging.version import Version

        # The "build" frontend will run setup.py twice within the same
        # temporary output directory. First for "sdist" and then for "wheel"
        # This would cause the build to fail with dirty "lib" and "include"
        # directories.
        # If the version in "libgpiod-version.txt" already matches our
        # requested tarball, then skip the fetch altogether.
        try:
            if open("libgpiod-version.txt").read() == LIBGPIOD_VERSION:
                log.info("skipping tarball fetch")
                return command(self, *args)
        except OSError:
            pass

        # Early exit for build tree with dirty lib/include dirs
        for check_dir in "lib", "include":
            if path.isdir(f"./{check_dir}"):
                raise BaseError(f"refusing to overwrite ./{check_dir}")

        with TemporaryDirectory(prefix="libgpiod-") as temp_dir:
            tarball_filename = TAR_FILENAME.format(version=LIBGPIOD_VERSION)
            tarball_url = f"{SRC_BASE_URL}{tarball_filename}"
            asc_url = f"{SRC_BASE_URL}{ASC_FILENAME}"

            log.info(f"fetching: {asc_url}")

            asc_filename, _ = urlretrieve(asc_url, path.join(temp_dir, ASC_FILENAME))

            tarball_sha256 = find_sha256sum(asc_filename, tarball_filename)

            if Version(LIBGPIOD_VERSION) < Version(LIBGPIOD_MINIMUM_VERSION):
                raise BaseError(f"requires libgpiod>={LIBGPIOD_MINIMUM_VERSION}")

            log.info(f"fetching: {tarball_url}")

            downloaded_tarball, _ = urlretrieve(
                tarball_url, path.join(temp_dir, tarball_filename)
            )

            log.info(f"verifying: {tarball_filename}")
            if sha256(downloaded_tarball) != tarball_sha256:
                raise BaseError(f"signature mismatch for {tarball_filename}")

            # Unpack the downloaded tarball
            log.info(f"unpacking: {tarball_filename}")
            with tarfile.open(downloaded_tarball) as f:
                if sys.version_info < (3, 12):
                    f.extractall(temp_dir)
                else:
                    f.extractall(temp_dir, filter=tarfile.fully_trusted_filter)

            # Copy the include and lib directories we need to build libgpiod
            base_dir = path.join(temp_dir, f"libgpiod-{LIBGPIOD_VERSION}")
            copytree(path.join(base_dir, "include"), "./include")
            copytree(path.join(base_dir, "lib"), "./lib")
            copy(path.join(base_dir, "LICENSES", "LGPL-2.1-or-later.txt"), "LICENSE")

        # Save the libgpiod version for sdist
        open("libgpiod-version.txt", "w").write(LIBGPIOD_VERSION)

        # Run the command
        return command(self, *args)

        # Clean up the build directory
        # rmtree("./lib", ignore_errors=True)
        # rmtree("./include", ignore_errors=True)
        # unlink("libgpiod-version.txt")
        # unlink("LICENSE")

    return wrapper


@fetch_tarball
def get_requires_for_build_sdist(config_settings=None):
    return _orig.get_requires_for_build_sdist(config_settings)


def build_sdist(sdist_dir, config_settings=None):
    filename = _orig.build_sdist(sdist_dir, config_settings)
    # Clean up the build directory
    rmtree("./lib", ignore_errors=True)
    rmtree("./include", ignore_errors=True)
    unlink("libgpiod-version.txt")
    unlink("LICENSE")
    return filename
