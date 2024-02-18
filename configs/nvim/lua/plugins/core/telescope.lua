--------------------------------------------------
---      telescope and it's extensions         ---
--------------------------------------------------
--- https://github.com/nvim-telescope/telescope.nvim
--- NOTE: Telescope's extensions are not installed here. Extensions are
--- just configed here.

-- https://github.com/nvim-telescope/telescope-project.nvim
-- Project manager.
local project = {
  order_by = "asc",
  search_by = "title",
  sync_with_nvim_tree = true,
}

-- fuzzy finder
return {
  "nvim-telescope/telescope.nvim",
  dependencies = {
    -- Some telescope extensions.
    { "nvim-telescope/telescope-live-grep-args.nvim" },
    { "xiyaowong/telescope-emoji" },
  },
  cmd = "Telescope",
  keys = {
    -- special
    { "<leader>,", "<cmd>Telescope buffers show_all_buffers=true<cr>", desc = "Buffers" },
    { "<leader><leader>", "<cmd>Telescope find_files<cr>", desc = "Find file" },
    { "<leader>fr", "<cmd>Telescope oldfiles<cr>", desc = "Recent files" },

    -- search
    { '<leader>s"', "<cmd>Telescope registers<cr>", desc = "Registers" },
    { "<leader>sa", "<cmd>Telescope autocommands<cr>", desc = "Auto commands" },
    { "<leader>sb", "<cmd>Telescope current_buffer_fuzzy_find<cr>", desc = "Current buffer" },
    { "<leader>sc", "<cmd>Telescope command_history<cr>", desc = "Command history" },
    { "<leader>sC", "<cmd>Telescope commands<cr>", desc = "Commands" },
    { "<leader>sd", "<cmd>Telescope diagnostics bufnr=0<cr>", desc = "Document diagnostics" },
    { "<leader>sD", "<cmd>Telescope diagnostics<cr>", desc = "Workspace diagnostics" },
    { "<leader>sk", "<cmd>Telescope keymaps<cr>", desc = "Key maps" },
    { "<leader>sM", "<cmd>Telescope man_pages<cr>", desc = "Man pages" },
    { "<leader>sm", "<cmd>Telescope marks<cr>", desc = "Marks" },
    { "<leader>so", "<cmd>Telescope vim_options<cr>", desc = "Options" },
    { "<leader>ss", "<cmd>Telescope lsp_document_symbols<cr>", desc = "Symbols" },
    { "<leader>sS", "<cmd>Telescope lsp_dynamic_workspace_symbols<cr>", desc = "Symbols(workspace)" },
    { "<leader>se", "<cmd>Telescope emoji<cr>", desc = "Emoji" },
    { "<leader>uC", "<cmd>Telescope colorscheme<cr>", desc = "Colorscheme" },
  },
  opts = {
    defaults = {
      prompt_prefix = " ",
      selection_caret = "❆ ",
      mappings = {
        i = {
          -- history
          ["<Down>"] = function(...)
            return require("telescope.actions").cycle_history_next(...)
          end,
          ["<Up>"] = function(...)
            return require("telescope.actions").cycle_history_prev(...)
          end,

          -- control preview
          ["<C-j>"] = function(...)
            return require("telescope.actions").preview_scrolling_down(...)
          end,
          ["<C-k>"] = function(...)
            return require("telescope.actions").preview_scrolling_up(...)
          end,
          ["<C-h>"] = function(...)
            return require("telescope.actions").preview_scrolling_left(...)
          end,
          ["<C-l>"] = function(...)
            return require("telescope.actions").preview_scrolling_right(...)
          end,

          -- split
          ["<C-x>"] = function(...)
            return require("telescope.actions").select_horizontal(...)
          end,
          ["<C-v>"] = function(...)
            return require("telescope.actions").select_vertical(...)
          end,
        },
        n = {
          ["q"] = function(...)
            return require("telescope.actions").close(...)
          end,
        },
      },
    },
    extensions = {
      project = project,
    },
  },
  config = function(_, opts)
    require("telescope").setup(opts)

    -- live_grep_args
    require("telescope").load_extension("live_grep_args")
    vim.keymap.set(
      "n",
      "<leader>/",
      require("telescope").extensions.live_grep_args.live_grep_args,
      { desc = "Live grep" }
    )
    local live_grep_args_shortcuts = require("telescope-live-grep-args.shortcuts")
    vim.keymap.set("n", "<leader>sw", function()
      require("telescope.builtin").grep_string({ word_match = "-w" })
    end, { desc = "Search word" })
    vim.keymap.set("v", "<leader>sv", live_grep_args_shortcuts.grep_visual_selection, { desc = "Search selection" })

    -- Project
    vim.keymap.set("n", "<leader>fp", require("telescope").extensions.project.project, { desc = "Projects" })
  end,
}
