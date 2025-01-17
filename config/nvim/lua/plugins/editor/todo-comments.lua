-- https://github.com/folke/todo-comments.nvim
-- Finds and lists all of the TODO, HACK, BUG, etc comment in your project and
-- loads them into a browsable list.

return {
  "folke/todo-comments.nvim",
  cmd = { "TodoTrouble", "TodoTelescope" },
  event = "LazyFile",
  config = true,
  keys = {
    -- stylua: ignore start
    { "]t",         function() require("todo-comments").jump_next() end, desc = "Next special comment" },
    { "[t",         function() require("todo-comments").jump_prev() end, desc = "Prev special comment" },
    { "<leader>sT", "<cmd>TodoTelescope keywords=TODO<cr>",              desc = "TODO" },
    { "<leader>xt", "<cmd>TodoTrouble keywords=TODO<cr>",                desc = "TODO" },
    { "<leader>xw", "<cmd>TodoTrouble keywords=WARN,WARNING,XXX<cr>",    desc = "WARN/WARNING/XXX" },
    { "<leader>xn", "<cmd>TodoTrouble keywords=NOTE,INFO<cr>",           desc = "NOTE/INFO" },
    { "<leader>xf", "<cmd>TodoTrouble keywords=FIX,FIXME,BUG,ISSUE<cr>", desc = "BUG/FIX/FIXME/ISSUE" },
    -- stylua: ignore end
  },
  opts = {
    keywords = { -- keywords recognized as todo comments
      FIX = {
        icon = " ", -- icon used for the sign, and in search results
        color = "error", -- can be a hex color, or a named color (see below)
        alt = { "FIXME", "BUG", "ISSUE" }, -- a set of other keywords that all map to this FIX keywords
      },
      TODO = { icon = " ", color = "info" },
      HACK = { icon = " ", color = "warning" },
      WARN = { icon = " ", color = "warning", alt = { "WARNING", "XXX", "WARN" } },
      PERF = { icon = "󰀝 ", alt = { "OPTIM", "PERFORMANCE", "OPTIMIZE" } },
      NOTE = { icon = " ", color = "hint", alt = { "INFO" } },
      TEST = { icon = "⏲ ", color = "test", alt = { "TESTING", "PASSED", "FAILED" } },
    },
  },
}
