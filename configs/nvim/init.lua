-- 1. Load config/options.lua
require("config.options")

-- 2. Install then load lazy.nvim
-- 3. Load all plugins which placed in lua/plugin by lazy.nvim just installed.
require("load_lazy")

-- 4. Load configs. Include keymaps, autocmds and others.
local Config = require("config")
-- Config.colorscheme = "edge"
Config.colorscheme = "dracula"
-- Config.colorscheme = "onedark"
-- Config.colorscheme = "dracula"
-- Config.colorscheme = "gruvbox"
-- Config.colorscheme = "tokyonight"
-- Config.colorscheme = "catppuccin"
Config.setup()
