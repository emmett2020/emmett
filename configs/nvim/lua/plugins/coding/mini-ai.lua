-- https://github.com/echasnovski/mini.ai
-- Better text-objects
-- The `a` means arround.
-- The `i` means inside.

return {
  "echasnovski/mini.ai",
  event = "VeryLazy",
  opts = {
    n_lines = 500, -- Number of lines within which textobject is searched
    custom_textobjects = nil,
  },
  config = function(_, opts)
    require("mini.ai").setup(opts)

    -- Register all keymaps when both mini.ai and which-key loaded.
    require("util").on_load("which-key.nvim", function()
      local a = {
        [" "] = "Whitespace",
        ['"'] = '"',
        ["'"] = "'",
        ["`"] = "`",
        ["("] = "(",
        [")"] = ")",
        [">"] = ">",
        ["<lt>"] = "<",
        ["]"] = "]",
        ["["] = "[",
        ["{"] = "{",
        ["}"] = "}",
        ["?"] = "User prompt",
        _ = "Underscore",
      }
      local i = vim.deepcopy(a)
      require("which-key").register({
        mode = { "o", "x" },
        i = i,
        a = a,
      })
    end)
  end,
}
