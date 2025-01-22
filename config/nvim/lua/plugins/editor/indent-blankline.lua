-- https://github.com/lukas-reineke/indent-blankline.nvim
-- indent guides for Neovim

return {
  "lukas-reineke/indent-blankline.nvim",
  main = "ibl",
  event = { "LazyFile" },
  opts = {
    indent = {
      char = "│",
      tab_char = "│",
    },

    -- PERF: for performance reason we disable scope
    -- Use mini.indentscope instead
    scope = { enabled = false },

    exclude = {
      filetypes = {
        "help",
        "alpha",
        "dashboard",
        "neo-tree",
        "Trouble",
        "trouble",
        "lazy",
        "mason",
        "notify",
        "toggleterm",
        "lazyterm",
      },
    },
  },

}
