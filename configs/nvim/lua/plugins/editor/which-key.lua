-- The which-key helps you remember key bindings by showing a popup
-- with the active keybindings of the command you started typing.
-- https://github.com/folke/which-key.nvim
return {
  "folke/which-key.nvim",
  event = "VeryLazy",
  opts = {
    plugins = { spelling = true },
    defaults = {
      mode = { "n", "v" },
      ["g"] = { name = "+goto" },
      ["gz"] = { name = "+surround" },
      ["]"] = { name = "+next" },
      ["["] = { name = "+prev" },
      ["<leader><tab>"] = { name = "+tabs" },
      ["<leader>b"] = { name = "+buffer" },
      ["<leader>c"] = { name = "+code" },
      ["<leader>f"] = { name = "+file/find" },
      ["<leader>g"] = { name = "+git" },
      ["<leader>gh"] = { name = "+hunks" },
      ["<leader>q"] = { name = "+quit/session" },
      ["<leader>s"] = { name = "+search" },
      ["<leader>u"] = { name = "+ui" },
      ["<leader>w"] = { name = "+windows" },
      ["<leader>x"] = { name = "+diagnostics/quickfix" },
      ["<leader>sn"] = { name = "+noice" },
      ["<leader>d"] = { name = "+debug" },
      ["<leader>da"] = { name = "+adapters" },
    },
  },
  config = function(_, opts)
    local wk = require("which-key")
    wk.setup(opts)
    wk.register(opts.defaults)
  end,
}
