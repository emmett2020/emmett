-- https://github.com/folke/which-key.nvim
-- The which-key helps you remember key bindings by showing a popup
-- with the active keybindings of the command you started typing.

return {
  "folke/which-key.nvim",
  event = "VeryLazy",
  opts = {
    defaults = {
      mode = { "n", "v" },
      ["g"] = { name = "+goto" },
      ["gz"] = { name = "+surround" },
      ["]"] = { name = "+next" },
      ["["] = { name = "+prev" },
      ["<leader><tab>"] = { name = "+tabs" },
      ["<leader>b"] = { name = "+buffer" },
      ["<leader>c"] = { name = "+code" },
      ["<leader>d"] = { name = "+debug" },
      ["<leader>da"] = { name = "+adapters" },
      ["<leader>f"] = { name = "+file" },
      ["<leader>fe"] = { name = "+explorer options" },
      ["<leader>g"] = { name = "+git" },
      ["<leader>gh"] = { name = "+hunks" },
      ["<leader>ghp"] = { name = "" },
      ["<leader>go"] = { name = "+options" },
      ["<leader>gob"] = { name = "" }, -- We find that the group needs at least one item, otherwise the group name will not take effect.
      ["<leader>n"] = { name = "+noice" },
      ["<leader>o"] = { name = "+options" },
      ["<leader>q"] = { name = "+quit/session" },
      ["<leader>s"] = { name = "+search" },
      ["<leader>u"] = { name = "+ui" },
      ["<leader>w"] = { name = "+windows" },
      ["<leader>x"] = { name = "+diagnostics/comments" },
    },
  },
  config = function(_, opts)
    local wk = require("which-key")
    wk.setup(opts)
    wk.register(opts.defaults)
  end,
}
