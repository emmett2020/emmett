-- 1. Load config/options.lua at first
require("config.options")

-- 2. Install and load lazy.nvim
--    Load all plugins which placed in lua/plugin by lazy.nvim just installed.
require("lazy_load")

-- 3. neovide config
if vim.g.neovide then
  require("neovide_config")
end

-- 4. Load configs. Include keymaps, autocmds and others.
local Config = require("config")
Config.colorscheme = "pastelnight"
-- Config.colorscheme = "edge"
Config.setup()
