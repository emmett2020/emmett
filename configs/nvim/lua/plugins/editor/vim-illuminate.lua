-- https://github.com/RRethy/vim-illuminate
-- Automatically highlights other instances of the word under your cursor.
-- This works with LSP, Treesitter, and regexp matching to find the other
-- instances.

return {
  "RRethy/vim-illuminate",
  event = { "BufReadPost", "BufNewFile" },
  opts = {
    delay = 200,
  },
  config = function(_, opts)
    require("illuminate").configure(opts)
  end,
}
