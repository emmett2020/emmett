-- 1. Load config/options.lua at first
require("config.options")

require("core.lazy_file").setup()
-- 2. Install and load lazy.nvim
-- 3. Load all plugins which placed in lua/plugin by lazy.nvim just installed.
require("lazy_load")


-- 4. Load configs. Include keymaps, autocmds and others.
local Config = require("config")
Config.colorscheme = "pastelnight"
Config.setup()

-- 5. Other DIY plugins.
require("util/hourly_notify").start_report()
