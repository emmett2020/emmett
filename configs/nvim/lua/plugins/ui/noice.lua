-- https://github.com/folke/noice.nvim
-- Highly experimental plugin that completely replaces the UI for messages, cmdline and the popupmenu
-- Noice.nvim needs nui.nvim and nvim-notify.

-- Add new routes here.
local routes = {
  -- 1. Adjust file saved hint to mini view.
  {
    view = "mini",               -- "mini" show message in the right bottom corner
    filter = {                   -- uses filters to route messages to specific views
      event = "msg_show",
      any = {                    -- match on item is ok
        { find = "%d+L, %d+B" }, --  equals to lua string.find
        { find = "; after #%d+" },
        { find = "; before #%d+" },
      },
    },
  },
  -- 2. Format hint will be shown in mini view.
  {
    view = "mini",
    filter = {
      event = "notify",
      find = "Formatting this file uses ",
    },
  },
}

return {
  "folke/noice.nvim",
  event = "VeryLazy",
  opts = {
    presets = {                     -- preset for eaiser configuration
      bottom_search = false,        -- use a classic bottom cmdline for search
      command_palette = true,       -- position the cmdline and popupmenu together
      long_message_to_split = true, -- long messages will be sent to a split
      inc_rename = true,            -- enables an input dialog for inc-rename.nvim
      lsp_doc_border = true,        -- add a border to hover docs and signature help
    },

    cmdline = {
      format = {
        search_down = { icon = "" },
        search_up = { icon = "" },
      },
    },

    lsp = {
      override = {
        ["vim.lsp.util.convert_input_to_markdown_lines"] = true,
        ["vim.lsp.util.stylize_markdown"] = true,
        ["cmp.entry.get_documentation"] = true, -- requires hrsh7th/nvim-cmp
      },
    },
    routes = routes,
  },
  -- stylua: ignore
  keys = {
    { "<leader>nl", function() require("noice").cmd("last") end,    desc = "Noice last message" },
    { "<leader>nh", function() require("noice").cmd("history") end, desc = "Noice history" },
    { "<leader>nd", function() require("noice").cmd("dismiss") end, desc = "Dismiss all" },
  },
}
