-- https://github.com/folke/which-key.nvim
-- The which-key helps you remember key bindings by showing a popup
-- with the active keybindings of the command you started typing.

return {
  "folke/which-key.nvim",
  event = "VeryLazy",
  opts = {
    spec = {
      mode = { "n", "v" },
      { "g",             group = "goto" },
      { "gz",            group = "surround" },
      { "]",             group = "next" },
      { "[",             group = "prev" },
      { "<leader><tab>", group = "tabs" },
      { "<leader>b",     group = "buffer" },
      { "<leader>c",     group = "code" },
      { "<leader>d",     group = "debug" },
      { "<leader>da",    group = "adapters" },
      { "<leader>f",     group = "file" },
      { "<leader>fe",    group = "explorer options" },
      { "<leader>g",     group = "git" },
      { "<leader>gh",    group = "hunks" },
      { "<leader>ghp",   group = "" },
      { "<leader>go",    group = "options" },
      { "<leader>gob",   group = "" }, -- We find that the group needs at least one item, otherwise the group name will not take effect.
      { "<leader>n",     group = "noice" },
      { "<leader>o",     group = "options" },
      { "<leader>q",     group = "quit/session" },
      { "<leader>s",     group = "search" },
      { "<leader>t",     group = "terminal" },
      { "<leader>u",     group = "ui" },
      { "<leader>w",     group = "windows" },
      { "<leader>x",     group = "diagnostics/comments" },
    },
  },
  config = function(_, opts)
    local wk = require("which-key")
    wk.setup(opts)
  end,
}
