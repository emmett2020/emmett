-- https://github.com/nvim-lualine/lualine.nvim
-- Depending on the location, the lualine plugin supports three status bars.
-- The top one is called tabline, the one next to the top is called winbar, and
-- the one at the bottom is called lualine.
-- Use statusline and tabline but not winbar.
-- Lualine has sections as shown below.
-- +-------------------------------------------------+
-- | A | B | C                             X | Y | Z |
-- +-------------------------------------------------+

local Config = require("config")

-- We should add some custom keymap.
local function GotoBuffer(prev)
  local B = require("lualine.components.buffers")
  if #B.bufpos2nr <= 1 then
    return
  end

  -- Current buffer and index
  local buf_num = vim.api.nvim_get_current_buf()
  local buf_idx = -1
  for idx, num in ipairs(B.bufpos2nr) do
    if buf_num == num then
      buf_idx = idx
    end
  end

  if prev then
    buf_idx = buf_idx - 1
    if buf_idx < 1 then
      buf_idx = #B.bufpos2nr
    end
  else
    buf_idx = buf_idx + 1
    if buf_idx > #B.bufpos2nr then
      buf_idx = 1
    end
  end
  vim.api.nvim_set_current_buf(B.bufpos2nr[buf_idx])
end

local function PrevBuffer()
  GotoBuffer(true)
end

local function NextBuffer()
  GotoBuffer(false)
end

local function HideLualine()
  require("lualine").hide({
    place = { "statusline", "tabline", "winbar" },
    unhide = false,
  })
end

local function ShowLualine()
  require("lualine").hide({
    place = { "statusline", "tabline", "winbar" },
    unhide = true,
  })
end

return {
  "nvim-lualine/lualine.nvim",
  event = "VeryLazy",
  lazy = true,
  opts = function()
    local icons = Config.icons

    return {
      options = {
        theme = "auto",
        globalstatus = true,
        disabled_filetypes = { statusline = { "dashboard", "alpha" } },
      },

      -- Lualine (bottom line)
      sections = {
        lualine_a = { "mode" },
        lualine_b = {
          { "branch", separator = "" },
          {
            "diff",
            symbols = { added = icons.git.added, modified = icons.git.modified, removed = icons.git.removed },
          },
        },
        lualine_c = {},
        lualine_x = {
          { "filename", path = 1, symbols = { modified = "  ", readonly = "", unnamed = "" }, separator = "" },
        },
        lualine_y = {},
        lualine_z = {
          { "progress", separator = "|", padding = { left = 1, right = 1 } },
          { "location", separator = "|", padding = { left = 1, right = 1 } },
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
    vim.keymap.set({ "n", "v" }, "<leader>bh", PrevBuffer, { desc = "Prev buffer" })
    vim.keymap.set({ "n", "v" }, "<leader>bl", NextBuffer, { desc = "Next buffer" })
    vim.keymap.set({ "n", "v" }, "[b", PrevBuffer, { desc = "Prev buffer" })
    vim.keymap.set({ "n", "v" }, "]b", NextBuffer, { desc = "Next buffer" })

    -- We hide lualine first and show it after enter a buffer
    -- Otherwise it will shown at Alpha plugin and netrw plugin.
    HideLualine()
    vim.api.nvim_create_autocmd({ "BufEnter" }, {
      pattern = "*",
      callback = ShowLualine,
    })
  end,
}
