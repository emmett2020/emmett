-- https://github.com/rafamadriz/friendly-snippets
-- friendly-snippets: It contains several snippets in different
-- languages, which are put into the json document. This plugin requires a
-- snippet engine to load it, and we use LuaSnip as the loading engine.

-- https://github.com/L3MON4D3/LuaSnip
-- LuaSnip: Snippet engine for Neovim which supports various filetype. This
-- plugin finally used by nvim-cmp.
-- NOTE: LuaSnip cannot be used directly by nvim-cmp, which relies on the
-- saadparwaiz1/cmp_luasnip plugin. The installation code for the cmp_luasnip
-- plugin is placed together with the installation code for nvim-cmp.

return {
  "L3MON4D3/LuaSnip",
  build = (not jit.os:find("Windows"))
      and "echo 'NOTE: jsregexp is optional, so not a big deal if it fails to build'; make install_jsregexp"
    or nil,
  dependencies = {
    "rafamadriz/friendly-snippets",
    config = function()
      -- Suggetsted usage by friendly-snippets
      require("luasnip.loaders.from_vscode").lazy_load()
    end,
  },
  opts = {
    -- This is just to ensure backwards-compatibility. i.g.: <shift><tab> should work.
    history = true,

    delete_check_events = "TextChanged",
  },
  keys = {
    {
      "<tab>",
      function()
        return require("luasnip").jumpable(1) and "<Plug>luasnip-jump-next" or "<tab>"
      end,
      expr = true,
      silent = true,
      mode = "i",
    },
    {
      "<tab>",
      function()
        require("luasnip").jump(1)
      end,
      mode = "s",
    },
    {
      "<s-tab>",
      function()
        require("luasnip").jump(-1)
      end,
      mode = { "i", "s" },
    },
  },
}
