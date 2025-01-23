# Introduction:
# This file is used to optimize the symbols of shared library. Generally
# speaking, if we are building a shared library, we should strip the .symtab to
# make SO cleaning. This function will generate two files, one is original SO
# while the other is debug symbol file.

# Parameters:
# 1. target_to_build: This is the target name to build.
# 2. target_lib_name: The full name of shared library whose .symtab will be stripped. e.g. libxxx.so.
# 3. target_lib_dir:  The absolute directory of `${target_lib_name}`.
# 4. debug_symbols_dir: the directory to install debug symbol file.

function (reserve_debug_symbol_then_strip_symtab
          target_to_build
          target_lib_name
          target_lib_dir
          debug_symbol_dir)
  # Only needed in "Release" build mode.
  if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
      set(target_lib_path "${target_lib_dir}/${target_lib_name}")
      set(debug_symbol_filename "${target_lib_name}.sym")
      set(debug_symbol_path "${debug_symbol_dir}/${debug_symbol_filename}")

      execute_process(COMMAND mkdir -p ${debug_symbol_dir})

      # Strip .symtab
      add_custom_command(TARGET ${target_to_build} POST_BUILD
          DEPENDS ${target_to_build}
          COMMENT "Stripping symbol of ${target_lib_path}"
          COMMAND sh -c
             "if objdump -h ${target_lib_path} | grep -q '.gnu_debuglink'; then \
                echo '${target_lib_path} unchanged, no need to strip'; \
              else \
                if [ -f ${debug_symbol_path} ]; then \
                  rm ${debug_symbol_path}; \
                fi; \
                objcopy --only-keep-debug ${target_lib_path} ${debug_symbol_path}; \
                objcopy --strip-all ${target_lib_path} ; \
                objcopy --add-gnu-debuglink=${debug_symbol_filename} ${target_lib_path}; \
              fi"
              WORKING_DIRECTORY ${debug_symbol_dir}
          VERBATIM)

      # Additional clean debug symbol file by "ninja clean".
      set_property(TARGET ${target_to_build} APPEND PROPERTY ADDITIONAL_CLEAN_FILES ${debug_symbol_path})
  endif()
endfunction(reserve_debug_symbol_then_strip_symtab)
