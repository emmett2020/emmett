-- https://github.com/akinsho/bufferline.nvim
-- This is what powers DailyVim's fancy-looking
-- tabs, which include filetype icons and close buttons.

-- WARNING: This plugin and keymaps are not used.
-- Use lualine.nvim instead since it has a more beautiful UI.

local Config = require("config")

return {
  "akinsho/bufferline.nvim",
  enable = false,
  event = "VeryLazy",
  keys = {
    { "<leader>bp", "<cmd>BufferLineTogglePin<cr>",            desc = "Pin Buffer" },
    { "<leader>bP", "<cmd>BufferLineGroupClose ungrouped<cr>", desc = "Delete non-pinned buffers" },
    { "[b",         "<cmd>BufferLineCyclePrev<cr>",            desc = "Prev buffer" },
    { "]b",         "<cmd>BufferLineCycleNext<cr>",            desc = "Next buffer" },
    { "<leader>bh", "<cmd>BufferLineCyclePrev<cr>",            desc = "Prev buffer" },
    { "<leader>bl", "<cmd>BufferLineCycleNext<cr>",            desc = "Next buffer" },
  },
  opts = {
    options = {
      -- Mouse is not needed.
      hover = {
        enable = false,
      },

      -- Separator style.
      separator_style = "thin",

      -- Allows highlight groups to be overriden.
      themable = true,

      -- The number is displayed on the left side of the label, and the height
      -- of the number can be adjusted by using raise and low.
      numbers = function(opts)
        return string.format("%s", opts.ordinal)
      end,

      -- you will get an indicator in the bufferline for a given tab if it has
      -- any errors This will allow you to tell at a glance if a particular
      -- buffer has errors.
      diagnostics = "nvim_lsp",
      diagnostics_indicator = function(_, _, diag)
        local icons = Config.icons.diagnostics
        local ret = (diag.error and icons.Error .. diag.error .. " " or "")
            .. (diag.warning and icons.Warn .. diag.warning or "")
        return vim.trim(ret)
      end,

      always_show_bufferline = false,

      -- No mouse needed, so does this option.
      show_buffer_close_icons = false,

      -- You can prevent the bufferline drawing above a *vertical* sidebar
      -- split such as a file explorer.
      offsets = {
        {
          filetype = "neo-tree",
          text = "Neo-tree",
          highlight = "Directory",
          text_align = "left",
        },
      },
    },
  },
}
