-- Git blame plugin for Neovim written in Lua.
-- https://github.com/f-person/git-blame.nvim
return {
  "f-person/git-blame.nvim",
  lazy = false,
  config = function(_, opts)
    require('gitblame').setup {
    enabled = false,
  }
  end,
  -- stylua: ignore
  keys = {
    {"<leader>gb", "<cmd>GitBlameToggle<cr>", desc = "Toggle Git blame", mode = {'n'}},
  },
}
