-- https://github.com/folke/flash.nvim
-- Flash enhances the built-in search functionality by showing labels location.
-- At the end of each match, letting you quickly jump to.
return {
  "folke/flash.nvim",
  event = "VeryLazy",
  ---@type Flash.Config
  opts = {},
  keys = {
    -- stylua: ignore start
    { "s", mode = { "n", "x", "o" }, function() require("flash").jump() end,       desc = "Flash" },
    { "S", mode = { "n", "o", "x" }, function() require("flash").treesitter() end, desc = "Flash Treesitter" },
    -- stylua: ignore end
  },
}
