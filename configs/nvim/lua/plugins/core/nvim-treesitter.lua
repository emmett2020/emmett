-- https://github.com/nvim-treesitter/nvim-treesitter
-- https://github.com/nvim-treesitter/nvim-treesitter-textobjects
-- nvim-treesitter is a new parser generator tool that we can use in Neovim to
-- power faster and more accurate syntax highlighting.

return {
  "nvim-treesitter/nvim-treesitter",
  version = "*",
  build = ":TSUpdate",
  -- event = { "BufReadPost", "BufNewFile" },
  event = { "LazyFile" },
  dependencies = {
    "nvim-treesitter/nvim-treesitter-textobjects",
  },
  cmd = { "TSUpdateSync" },

  opts = {
    highlight = { enable = true },
    indent = { enable = true },
    ensure_installed = {
      "bash",
      "c",
      "cpp",
      "lua",
      "python",
      "yaml",
      "markdown",
      "vim",
      "vimdoc",
    },

    -- See nvim-treesitter-textobjects for more choices.
    textobjects = {
      select = {
        enable = true,
        lookahead = true,
        keymaps = { -- select according to textobjects
          ["ac"] = { query = "@class.outer", desc = "Class" },
          ["ic"] = { query = "@class.inner", desc = "Class" },
          ["af"] = { query = "@function.outer", desc = "Function" },
          ["if"] = { query = "@function.inner", desc = "Function" },
        },
      },
      move = { -- move according to textobjects
        enable = true,
        set_jumps = true,
        goto_next_start = {
          ["]c"] = { query = "@class.outer", desc = "Next class start" },
          ["]f"] = { query = "@function.outer", desc = "Next function start" },
          ["]z"] = { query = "@fold", query_group = "folds", desc = "Next fold" },
        },
        goto_next_end = {
          ["]C"] = { query = "@class.outer", desc = "Next class end" },
          ["]F"] = { query = "@function.outer", desc = "Next function end" },
        },
        goto_previous_start = {
          ["[c"] = { query = "@class.outer", desc = "Prev class start" },
          ["[f"] = { query = "@function.outer", desc = "Prev function start" },
        },
        goto_previous_end = {
          ["[C"] = { query = "@class.outer", desc = "Prev class end" },
          ["[F"] = { query = "@function.outer", desc = "Prev function end" },
        },
      },
    },
  },

  config = function(_, opts)
    require("nvim-treesitter.configs").setup(opts)
    require("util").on_load("which-key.nvim", function()
      local a = {
        ["c"] = "Class",
        ["f"] = "Function",
      }
      local i = {
        ["c"] = "Class",
        ["f"] = "Function",
      }
      require("which-key").register({
        mode = { "o", "x" },
        i = i,
        a = a,
      })
    end)
  end,
}
