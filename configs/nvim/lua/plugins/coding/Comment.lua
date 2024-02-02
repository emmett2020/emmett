-- https://github.com/numToStr/Comment.nvim

return {
  "numToStr/Comment.nvim",
  opts = {
    -- Disable all shortcuts provided by Comment.nvim
    mappings = {
      basic = false,
      extra = false,
    },
  },

  -- We hope that the comment function is very simple.
  -- We can enter the <leader>cc command for different filetypes and modes,
  -- and then comment and uncomment the contents of some lines.
  keys = {
    {
      "<leader>cc",
      function()
        return vim.api.nvim_get_vvar("count") == 0 and "<Plug>(comment_toggle_linewise_current)"
          or "<Plug>(comment_toggle_linewise_count)"
      end,
      desc = "Comment",
      expr = true,
      silent = true,
      mode = { "n" },
    },
    {
      "<leader>cc",
      "<Plug>(comment_toggle_linewise_visual)",
      desc = "Comment",
      silent = true,
      mode = { "x" },
    },
  },
  lazy = false,
}
