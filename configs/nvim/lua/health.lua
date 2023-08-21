---------------------------------------------------------------------
---            Validate version and check envrionment             ---
---------------------------------------------------------------------

local M = {}

-- Options
local nvim_version = "nvim-0.8.0"
local lazy_version = ">=9.1.0"

-- commands_name, minimal_version, maximum_version
-- Commands that must exist in envrioments.
local required_commands = {
  { "git", "", "" },
  { "rd", "", "" },
  { "lazygit", "", "" },
  { "fd", "", "" },
}

-- Commands that should exist in envrionments.
local optional_commands = {}

-- For log
local start = vim.health.start or vim.health.report_start
local ok = vim.health.ok or vim.health.report_ok
local warn = vim.health.warn or vim.health.report_warn
local error = vim.health.error or vim.health.report_error

-- Check version of lazy.nvim.
function M.check_lazy_nvim_version()
  local Semver = require("lazy.manage.semver")
  local installed_version = require("lazy.core.config").version or "0.0.0"
  return Semver.range(lazy_version):matches(installed_version)
end

-- Check version of neovim.
function M.check_neovim_version()
  if vim.fn.has(nvim_version) == 1 then
    ok("Using " .. nvim_version)
  else
    error(nvim_version .. " is required")
  end
end

-- Check version of a specific command.
function M.check_specific_command_version(command, min_verion, max_version) end

-- Check necessary commands.
function M.check_commands()
  for _, cmd in ipairs(required_commands) do
    -- FIXME
    local name = type(cmd) == "string" and cmd or vim.inspect(cmd)
    local commands = type(cmd) == "string" and { cmd } or cmd
    ---@cast commands string[]
    local found = false

    for _, c in ipairs(commands) do
      if vim.fn.executable(c) == 1 then
        name = c
        found = true
      end
    end

    if found then
      ok(("`%s` is installed"):format(name))
    else
      warn(("`%s` is not installed"):format(name))
    end
  end
end

function M.check()
  M.check_neovim_version()
  M.check_lazy_nvim_version()
  M.check_commands()
end

return M
