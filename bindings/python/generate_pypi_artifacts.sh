#!/usr/bin/env sh

if [ -f /etc/os-release ]; then
    . /etc/os-release
fi

# Respect environment override of binding library version
LIBGPIOD_VERSION=${LIBGPIOD_VERSION:-2.1.2}

# We don't need a specific version of python, but we do need it
if ! command -v python3 >/dev/null 2>&1; then
    echo "Python3 is required to generate PyPI artifacts"
    exit 1
fi

# For wheel builds, we need docker, pip + virtualenv, and qemu-user static
if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required to generate wheels"
    exit 1
fi

if ! $(python3 -m pip -h >/dev/null 2>&1); then
    echo "pip is required to generate wheels"
    exit 1
fi

# Check for a virtual environment tool
has_venv=$(python3 -m venv -h >/dev/null 2>&1 && echo 1 || echo 0)
has_virtualenv=$(python3 -m virtualenv -h >/dev/null 2>&1 && echo 1 || echo 0)

if ! ([ $has_venv ] || [ $has_virtualenv ]); then
    echo "A virtual environment tool is required to generate wheels"
    exit 1
fi

# If we're on a Debian based system, make assumptions about binfmt paths
if [ ${ID_LIKE} == "debian" ]; then
    echo "You're on Debian"
fi

venv_module=$([ $has_virtualenv ] && echo "virtualenv" || echo "venv" )

echo ${has_venv}
echo ${has_virtualenv}
echo ${venv_module}
echo ${LIBGPIOD_VERSION}

# stage the build in a temp directory
src_dir=$(pwd)
temp_dir=$(mktemp -d)
cd $temp_dir
python3 -m $venv_module .venv

venv_python="${temp_dir}/.venv/bin/python3"
$venv_python -m pip install build==1.2.1 cibuildwheel==2.18.1

LIBGPIOD_VERSION=${LIBGPIOD_VERSION} $venv_python -m build --sdist --outdir ./dist $src_dir 
sdist=$(find ./dist -name '*.tar.gz')

$venv_python -m cibuildwheel --platform linux --archs x86_64 $sdist --output-dir dist/
cp -ra dist/ $src_dir/

cd $src_dir
rm -rf $temp_dir