-- https://github.com/echasnovski/mini.comment

return {
  "echasnovski/mini.comment",
  event = "VeryLazy",
  opts = {
    options = {
      custom_commentstring = function()
        local commentstring = require("ts_context_commentstring.internal")
        return commentstring.calculate_commentstring() or vim.bo.commentstring
      end,

      -- Use `''` (empty string) to disable one.
      mappings = {
        -- Toggle comment (like `gcip` - comment inner paragraph)
        -- for both Normal and Visual modes
        comment = "gc",

        -- Toggle comment on current line
        comment_line = "gcc",

        -- Define 'comment' textobject (like `dgc` - delete whole comment block)
        textobject = "gc",
      },
    },
  },
}
