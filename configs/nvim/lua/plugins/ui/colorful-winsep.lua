-- https://github.com/nvim-zh/colorful-winsep.nvim
-- This plugin will color the border of active window

return {
  "nvim-zh/colorful-winsep.nvim",
  config = true,
  event = { "WinNew" },
  opts = {
    highlight = {
      fg = "#C0C0C0",
    },
  },
}
