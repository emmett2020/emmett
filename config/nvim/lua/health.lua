---------------------------------------------------------------------
---            Validate version and check envrionment             ---
---------------------------------------------------------------------

-- For log
local ok = vim.health.ok
local warn = vim.health.warn
local error = vim.health.error

-- Check all the necessary plugins and report error if no exists.
local M = {}

local nvim_version = "nvim-0.10.0"

-- commands_name, minimal_version, maximum_version
-- Commands that must exist in envrioments.
local required_commands = {
  { "git",     "", "" },
  { "rg",      "", "" },
  { "lazygit", "", "" },
  { "fd",      "", "" },
}

-- Check version of neovim.
function M.check_neovim_version()
  if vim.fn.has(nvim_version) == 1 then
    ok("Using " .. nvim_version)
  else
    error(nvim_version .. " is required")
  end
end

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
  M.check_commands()
end

return M
