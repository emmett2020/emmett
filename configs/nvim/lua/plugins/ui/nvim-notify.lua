-- Better `vim.notify()`
local Util = require("util")
return {
  "rcarriga/nvim-notify",
  opts = {
    timeout = 1500,
    max_height = function()
      return math.floor(vim.o.lines * 0.75)
    end,
    max_width = function()
      return math.floor(vim.o.columns * 0.75)
    end,
    background_colour = "#000000",
  },
}
