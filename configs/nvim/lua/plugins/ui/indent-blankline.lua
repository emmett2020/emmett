-- https://github.com/lukas-reineke/indent-blankline.nvim
-- indent guides for Neovim

return {
  "lukas-reineke/indent-blankline.nvim",
  main = "ibl",
  event = { "BufReadPost", "BufNewFile" },
  config = function()
    require("ibl").setup({ indent = { char = "│" } })
  end,
}
