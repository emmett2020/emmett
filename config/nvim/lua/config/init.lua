--------------------------------------------------------------------
---               Load config files in config directory          ---
--------------------------------------------------------------------

local M = {}

---@param name "autocmds" | "options" | "keymaps" | "icons"
local function load_config_file(name)
  return require("config." .. name)
end

---@param colorscheme string|fun()
local function load_colorscheme(colorscheme)
  local LazyUtil = require("lazy.core.util")
  LazyUtil.try(function()
    if type(colorscheme) == "function" then
      colorscheme()
    else
      vim.cmd.colorscheme(colorscheme)
    end
  end, {
    msg = "Could not load colorscheme.",
    on_error = function(msg)
      LazyUtil.error(msg)
      vim.cmd.colorscheme("habamax")
    end,
  })
end

local defaults = {
  -- colorscheme can be a string like `catppuccin`
  -- or a function that will load the colorscheme
  ---@type string|fun()
  colorscheme = function()
    require("tokyonight").load()
  end,

  -- icons used by other plugins
  icons = load_config_file("icons"),
}

setmetatable(M, {
  __index = defaults,
})

function M.setup()
  -- 1. Delay notifications till vim.notify was replaced or after 500ms.
  require("util").lazy_notify()

  -- 2. Load autocmds and keymaps.
  load_config_file("autocmds")
  load_config_file("keymaps")

  -- 3. Load color scheme.
  load_colorscheme(M.colorscheme)
end

return M
