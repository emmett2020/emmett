------------------------------------------------------------
---      Install lazy.nvim and Load all plugins by it    ---
------------------------------------------------------------

-- 1. Install lazy.nvim at `lazy_path`.
local lazy_path = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not vim.loop.fs_stat(lazy_path) then
  vim.fn.system({
    "git",
    "clone",
    "--filter=blob:none",
    "https://github.com/folke/lazy.nvim.git",
    "--branch=stable",
    lazy_path,
  })
end
vim.opt.rtp:prepend(vim.env.LAZY or lazy_path)

-- 2. Load all plugins placed at `plugins`.
require("lazy").setup({
  spec = {
    { import = "plugins" },
  },
  defaults = {
    lazy = true, -- Plugins only be loaded when needed.
    version = false, -- Always use the latest git commit.
  },
  install = {
    -- Try to load one of these colorschemes
    -- when starting an installation during startup.
    colorscheme = { "tokyonight", "habamax" },
  },
  checker = { enabled = true }, -- Automatically check for plugin updates
  performance = {
    rtp = {
      -- disable some rtp plugins
      disabled_plugins = {
        "gzip",
        -- "matchit",
        -- "matchparen",
        -- "netrwPlugin",
        "tarPlugin",
        "tohtml",
        "tutor",
        "zipPlugin",
      },
    },
  },
})
