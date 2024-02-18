return {
  "akinsho/toggleterm.nvim",
  url = "git@github.com:akinsho/toggleterm.nvim.git",
  version = "*",
  config = function(_, opt)
    require("toggleterm").setup(opt)
  end,
}
