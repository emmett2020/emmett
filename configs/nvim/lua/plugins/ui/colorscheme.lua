return {
  -- tokyonight
  {
    "folke/tokyonight.nvim",
    lazy = true,
    opts = { style = "moon" },
  },

  -- catppuccin
  {
    "catppuccin/nvim",
    lazy = true,
    name = "catppuccin",
    opts = {
      integrations = {
        alpha = true,
        cmp = true,
        flash = true,
        gitsigns = true,
        illuminate = true,
        indent_blankline = { enabled = true },
        lsp_trouble = true,
        mason = true,
        mini = true,
        native_lsp = {
          enabled = true,
          underlines = {
            errors = { "undercurl" },
            hints = { "undercurl" },
            warnings = { "undercurl" },
            information = { "undercurl" },
          },
        },
        navic = { enabled = true, custom_bg = "lualine" },
        neotest = true,
        noice = true,
        notify = true,
        neotree = true,
        semantic_tokens = true,
        telescope = true,
        treesitter = true,
        which_key = true,
      },
    },
  },

  -- Grouvbox
  {
    "morhetz/gruvbox",
    lazy = true,
    opts = {},
    config = function() end,
  },

  -- darcula
  {
    "Mofiqul/dracula.nvim",
    lazy = true,
  },

  -- edge
  {
    "sainnhe/edge",
    lazy = true,
  },

  -- nightfox
  {
    "EdenEast/nightfox.nvim",
    lazy = true,
  },

  -- nord
  {
    "shaunsingh/nord.nvim",
    lazy = true,
  },

  -- onedark
  {
    "navarasu/onedark.nvim",
    lazy = true,
  },

  -- onenord
  {
    "rmehri01/onenord.nvim",
    lazy = true,
  },

  -- everforest
  {
    "sainnhe/everforest",
    lazy = true,
  },

  -- aurora
  {
    "ray-x/aurora",
    lazy = true,
  },
}
