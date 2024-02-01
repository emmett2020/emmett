-- https://github.com/nvim-zh/colorful-winsep.nvim
return {
  "nvim-zh/colorful-winsep.nvim",
  config = true,
  event = { "WinNew" },
  opts = {
    highlight = {
      -- fg = "#E6E8FA",
      fg = "#C0C0C0",
      -- fg = "#1F3442",
    },
  },
}
