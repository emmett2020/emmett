-- https://github.com/folke/lazy.nvim

return {
  "folke/lazy.nvim",
  version = "*",
  opt = {
    ui = {
      border = "rounded",
    },
  },
  -- Enable profiling of lazy.nvim. This will add some overhead,
  -- so only enable this when you are debugging lazy.nvim
  profiling = {
    -- Enables extra stats on the debug tab related to the loader cache.
    -- Additionally gathers stats about all package.loaders
    loader = true,
    -- Track each new require in the Lazy profiling tab
    require = true,
  },
}
