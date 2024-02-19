-- https://github.com/nvim-treesitter/nvim-treesitter
-- https://github.com/nvim-treesitter/nvim-treesitter-textobjects
-- nvim-treesitter is a new parser generator tool that we can use in Neovim to
-- power faster and more accurate syntax highlighting.

return {
  "nvim-treesitter/nvim-treesitter",
  version = "*",
  build = ":TSUpdate",
  event = { "LazyFile", "VeryLazy" },
  dependencies = {
    "nvim-treesitter/nvim-treesitter-textobjects",
  },
  cmd = { "TSUpdateSync", "TSUpdate", "TSInstall" },
  init = function(plugin)
    -- Referenced by LazyVim@10.x @folke
    -- PERF: add nvim-treesitter queries to the rtp and it's custom query predicates early
    -- This is needed because a bunch of plugins no longer `require("nvim-treesitter")`, which
    -- no longer trigger the **nvim-treeitter** module to be loaded in time.
    -- Luckily, the only thins that those plugins need are the custom queries, which we make available
    -- during startup.
    require("lazy.core.loader").add_to_rtp(plugin)
    require("nvim-treesitter.query_predicates")
  end,
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
      -- PERF: currently "select" has very slow startup perfomance.
      -- move according to textobjects
      move = {
        enable = true,
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
  end,
}
