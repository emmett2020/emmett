-- https://github.com/nvim-lualine/lualine.nvim
-- Depending on the location, the lualine plugin supports three status bars.
-- The top one is called tabline, the one next to the top is called winbar, and
-- the one at the bottom is called lualine.
-- Use statusline and tabline but not winbar.
-- Lualine has sections as shown below.
-- +-------------------------------------------------+
-- | A | B | C                             X | Y | Z |
-- +-------------------------------------------------+

return {
  "nvim-lualine/lualine.nvim",
  event = "VeryLazy",
  init = function()
    vim.g.lualine_laststatus = vim.o.laststatus
    if vim.fn.argc(-1) > 0 then
      vim.o.statusline = " "
    else
      vim.o.laststatus = 0
    end
  end,
  opts = function()
    -- PERF: we don't need this lualine require madness
    local lualine_require = require("lualine_require")
    lualine_require.require = require

    local icons = require("config").icons
    vim.o.laststatus = vim.g.lualine_laststatus

    return {
      options = {
        theme = "auto",
        globalstatus = true,
        disabled_filetypes = { statusline = { "dashboard", "alpha", "starter" } },
      },

      -- Lualine (bottom line)
      sections = {
        lualine_a = { "mode" },
        lualine_b = {
          { "branch", separator = "" },
          {
            "diff",
            symbols = { added = icons.git.added, modified = icons.git.modified, removed = icons.git.removed },
            source = function()
              local gitsigns = vim.b.gitsigns_status_dict
              if gitsigns then
                return {
                  added = gitsigns.added,
                  modified = gitsigns.changed,
                  removed = gitsigns.removed,
                }
              end
            end,
          },
        },
        lualine_c = {},
        lualine_x = {
          { "filename", path = 1, symbols = { modified = "  ", readonly = "", unnamed = "" }, separator = "" },
        },
        lualine_y = {},
        lualine_z = {
          { "progress", separator = "", padding = { left = 1, right = 0 } },
          { "location", separator = "", padding = { left = 0, right = 1 } },
        },
      },

      -- Tabline (top line)
      tabline = {
        lualine_a = { { "buffers", mode = 2 } },
        lualine_b = {},
        lualine_c = {},
        lualine_x = {},
        lualine_y = {
          {
            "diagnostics",
            symbols = {
              error = icons.diagnostics.Error,
              warn = icons.diagnostics.Warn,
              info = icons.diagnostics.Info,
              hint = icons.diagnostics.Hint,
            },
          },
        },
        lualine_z = {
          function()
            return " " .. os.date("%R")
          end,
        },
      },

      extensions = { "neo-tree", "lazy" },
    }
  end,

  config = function(_, opts)
    require("lualine").setup(opts)
  end,
}
