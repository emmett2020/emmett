#!/bin/bash
cat << END
This script supports compiling binary and running binary using different compiler versions.
That's to say, the version of compiler during compiling could greater than the
version of compiler during running.
|------------------------------|------------------------------|
|          item                |         explanation          |
|------------------------------|------------------------------|
|    needs root permission?    |              No              |
|------------------------------|------------------------------|
|          dependencies        |           lddtree            |
|                              |           patchelf           |
|------------------------------|------------------------------|
|          fellows             |         install.sh           |
|                              |         uninstall.sh         |
|------------------------------|------------------------------|
END

# Exit on error, treat unset variables as an error, and fail on pipeline errors
set -euo pipefail

CUR_SCRIPT_DIR=$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)
SCRIPTS_DIR="${CUR_SCRIPT_DIR}/../../../../.."

PROJECT_ROOT_PATH="${SCRIPTS_DIR}/.."
echo "PROJECT_ROOT_PATH: ${PROJECT_ROOT_PATH}"

PROJECT_BUILD_PATH="${PROJECT_ROOT_PATH}"/build
pushd "${PROJECT_BUILD_PATH}" &> /dev/null

BINARY_NAME="cpp-lint-action"
PACKAGE_NAME="cpp_lint_action"
INTERPRETER_INSTALL_PATH="/usr/local/lib/${BINARY_NAME}"

echo "Start to package ${BINARY_NAME}, distribution product name: ${PACKAGE_NAME}"
[[ -d "${PACKAGE_NAME}" ]] && rm -rf "${PACKAGE_NAME}"

echo "1. Start to collect all dependent shared libraries for ${BINARY_NAME}"
lddtree "${BINARY_NAME}" \
  --copy-to-tree "${PACKAGE_NAME}" \
  --libdir "/lib/${BINARY_NAME}" \
  --bindir "/bin"

# shellcheck disable=SC2010
# We know it's safe.
interpreter=$(ls ${PACKAGE_NAME}/lib/${BINARY_NAME} | grep "ld-linux")
echo "2. Start to set new interpreter path: ${INTERPRETER_INSTALL_PATH}/${interpreter}"
patchelf --set-interpreter \
  "${INTERPRETER_INSTALL_PATH}/${interpreter}" \
  ${PACKAGE_NAME}/bin/${BINARY_NAME}

echo "3. Start to set rpath for: ${BINARY_NAME}"
# shellcheck disable=SC2016
# ORIGIN shouldn't be translated while BINARY_NAME should be translated.
patchelf --force-rpath \
  --set-rpath \
  '$ORIGIN/../lib/'${BINARY_NAME} \
  "${PACKAGE_NAME}/bin/${BINARY_NAME}"

echo "4. Start to compress"
# -- ${PACKAGE_NAME}
# ------- install.sh
# ------- uninstall.sh
# ------- bin/
# ------- lib/
cp "${CUR_SCRIPT_DIR}"/install.sh ${PACKAGE_NAME}
cp "${CUR_SCRIPT_DIR}"/uninstall.sh ${PACKAGE_NAME}
tar -cvf ${PACKAGE_NAME}.tar.gz ${PACKAGE_NAME}

popd &> /dev/null
echo "Successfully packaged ${BINARY_NAME} and it's dependencies into ${PACKAGE_NAME}"
