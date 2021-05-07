-- https://github.com/folke/persistence.nvim
-- Session management.
-- This saves your session in the background,
-- keeping track of open buffers, window arrangement, and more.
-- You can restore sessions when returning through the dashboard.
return {
  "folke/persistence.nvim",
  event = "BufReadPre",
  opts = {
    options = {
      "buffers",
      "curdir",
      "tabpages",
      "winsize",
      "help",
      "globals",
      "skiprtp",
    },

    -- Directory where session files are saved.
    dir = vim.fn.expand(vim.fn.stdpath("state") .. "/sessions/"),
  },

  -- stylua: ignore
  keys = {
    { "<leader>qs", function() require("persistence").load() end,                desc = "Restore session" },
    { "<leader>ql", function() require("persistence").load({ last = true }) end, desc = "Restore last session" },
    { "<leader>qd", function() require("persistence").stop() end,                desc = "Don't save current session" },
  },
}
