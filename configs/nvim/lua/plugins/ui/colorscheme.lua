-----------------------------------
-- Index     Name
--  01       tokyonight
--  02       catppuccin
--  03       gruvbox
--  04       dracula
--  05       edge
--  06       nightfox
--  07       nord
--  08       onedark
--  09       onenord
--  10       everforest
--  11       aurora
--  12       kanagawa
--  13       pastelnight
--  14       tol
--  15       sweetie
--  16       bluloco
-----------------------------------

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

  -- gruvbox
  {
    "morhetz/gruvbox",
    lazy = true,
    opts = {},
    config = function() end,
  },

  -- dracula
  {
    "Mofiqul/dracula.nvim",
    lazy = true,
    opts = {
      transparent_bg = false,
    },
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

  -- kanagawa
  {
    "rebelot/kanagawa.nvim",
    lazy = true,
  },

  -- pastelnight
  {
    "pauchiner/pastelnight.nvim",
    lazy = true,
    opts = {},
  },

  -- tol.nvim
  {
    "dustypomerleau/tol.nvim",
    lazy = true,
  },

  -- sweetie
  {
    "NTBBloodbath/sweetie.nvim",
    lazy = true,
  },

  -- bluloco
  {
    "uloco/bluloco.nvim",
    lazy = true,
    dependencies = { "rktjmp/lush.nvim" },
  },

  {
    "eldritch-theme/eldritch.nvim",
    lazy = true,
    dependencies = { "rktjmp/lush.nvim" },
  },
}
