# C++ Project
## 1. Add a new C++ subproject
1. Create subproject in suitable directory, e.g. `cpp/bench/your_project_name`.
2. Copy `CMakeLists.txt` from existing subproject to the new subproject. Then modify `CMakeLists.txt` to enable your basic compilation workflow.
3. Add this new subproject into `.clangd` to enable `LSP` for `C++`.
4. Add this new subproject into `.github/workflows/ci_cpp.yml` to enable CI.
5. (Optional) Add a README.md to describe what this new subproject will do.

## 2. Perform formatting and code syntax checks before submit.
You need to install some tools to enable automatically source code static analysis and get all things passed before you submit a pull request.
|     Tool      |   Version   |
| ------------- | ----------- |
| clang-tidy    |   20.0.0+   |
| clang-format  |   20.0.0+   |

Instead of installing these tools directly, you can opt for the Language Server Protocol (LSP), which integrates these tools seamlessly. One of the most widely used `LSP` implementations is `clangd`, a part of the LLVM project. `clangd` offers features like code completion, go-to-definition, format checking, and code linting.

Using `clangd` is quite straightforward these days. If you're using `VSCode`, simply install the `clangd` VSCode Extension and configure it. For `Neovim` users, you can utilize `Mason` to download and set it up. While I haven't tested it on other IDEs, the process is likely similarly simple.

Ensure that your `clangd` version is 20.0.0 or higher for optimal functionality.

# Shell Scripts
## 1. Add a new shell script
Your scripts should include the following elements:
1. Proper "here document": Provide clear documentation at the beginning of the script, specifying whether root permissions are required or if there are any other limitations.
2. set -euo pipefail: Ensure robust error handling by including this line at the start of your script.
3. Your function: Define the core functionality of the script.

Each script should have a minimized and focused purpose. Avoid referencing other scripts unless absolutely necessary.
Once your script has been thoroughly tested and passes all checks, place it in the appropriate directory:
For x86-64 only scripts, place them in ubuntu/x86.
For scripts that support both x86 and ARM, place them directly in the ubuntu directory.
We prefer writing architecture-specific or OS-specific scripts even if it results in some code redundancy, as this approach ensures clarity and maintainability.

## 2. Perform formatting and code syntax checks before submit
|     Tool      |   Version   |
| ------------- | ----------- |
| shfmt         |   0.10.0+   |
| shellcheck    |   3.10.0+   |
