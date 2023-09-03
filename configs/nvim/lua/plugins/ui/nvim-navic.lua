-- lsp symbol navigation for lualine.
-- This shows where in the code structure you are
-- within functions, classes, etc - in the statusline.
-- https://github.com/SmiteshP/nvim-navic
local Util = require("util")
local Config = require("config")

return {
  "SmiteshP/nvim-navic",
  lazy = true,
  init = function()
    vim.g.navic_silence = true
    Util.on_attach(function(client, buffer)
      if client.server_capabilities.documentSymbolProvider then
        require("nvim-navic").attach(client, buffer)
      end
    end)
  end,
  opts = function()
    return {
      separator = " ",
      highlight = true,
      depth_limit = 0,
      depth_limit_indicator = "..",
      icons = Config.icons.kinds,
      -- Single click to goto element,
      -- double click to open nvim-navbuddy on the clicked element.
      click = true,
    }
  end,
}
