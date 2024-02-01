--------------------------------------------------
---      telescope and it's extensions         ---
--------------------------------------------------
---
--- !!!Note all telescope's extensions aren't installed here,
--- !!!     extensions are just configed here.
---

local Util = require("util")

-- Project manager.
-- https://github.com/nvim-telescope/telescope-project.nvim
local project = {
  order_by = "asc",
  search_by = "title",
  sync_with_nvim_tree = true,
}

-- fuzzy finder
-- https://github.com/nvim-telescope/telescope.nvim
return {
  "nvim-telescope/telescope.nvim",
  -- tag = "0.1.2",
  cmd = "Telescope",
  version = false,
  keys = {
    { "<leader>,", "<cmd>Telescope buffers show_all_buffers=true<cr>", desc = "File Buffers" },
    { "<leader>/", Util.telescope("live_grep", { cwd = false }), desc = "Search Content(cwd)" },
    { "<leader>:", "<cmd>Telescope command_history<cr>", desc = "Command History" },
    { "<leader><space>", Util.telescope("files", { cwd = false }), desc = "Find Files(cwd)" },

    { "<leader>fb", "<cmd>Telescope buffers<cr>", desc = "File Buffers" },
    -- find
    { "<leader>ff", Util.telescope("files", { cwd = false }), desc = "Find Files(cwd)" },
    { "<leader>fF", Util.telescope("files"), desc = "Find Files(root)" },
    { "<leader>fr", Util.telescope("oldfiles", { cwd = vim.loop.cwd() }), desc = "Recent Files(cwd)" },
    { "<leader>fR", "<cmd>Telescope oldfiles<cr>", desc = "Recent Files(root)" },

    -- git
    { "<leader>gc", "<cmd>Telescope git_commits<CR>", desc = "Commits" },
    { "<leader>gs", "<cmd>Telescope git_status<CR>", desc = "Status" },

    -- search
    { '<leader>s"', "<cmd>Telescope registers<cr>", desc = "Registers" },
    { "<leader>sa", "<cmd>Telescope autocommands<cr>", desc = "Auto Commands" },
    { "<leader>sb", "<cmd>Telescope current_buffer_fuzzy_find<cr>", desc = "Current Buffer" },
    { "<leader>sc", "<cmd>Telescope command_history<cr>", desc = "Command History" },
    { "<leader>sC", "<cmd>Telescope commands<cr>", desc = "Commands" },
    { "<leader>sd", "<cmd>Telescope diagnostics bufnr=0<cr>", desc = "Document Diagnostics" },
    { "<leader>sD", "<cmd>Telescope diagnostics<cr>", desc = "Workspace Diagnostics" },
    { "<leader>sg", Util.telescope("live_grep", { cwd = false }), desc = "Grep(cwd)" },
    { "<leader>sG", Util.telescope("live_grep"), desc = "Grep(root)" },
    { "<leader>sh", "<cmd>Telescope help_tags<cr>", desc = "Help Pages" },
    { "<leader>sH", "<cmd>Telescope highlights<cr>", desc = "Search Highlight Groups" },
    { "<leader>sk", "<cmd>Telescope keymaps<cr>", desc = "Key Maps" },
    { "<leader>sM", "<cmd>Telescope man_pages<cr>", desc = "Man Pages" },
    { "<leader>sm", "<cmd>Telescope marks<cr>", desc = "Marks" },
    { "<leader>so", "<cmd>Telescope vim_options<cr>", desc = "Options" },
    -- { "<leader>sR", "<cmd>Telescope resume<cr>", desc = "Resume" },
    { "<leader>sw", Util.telescope("grep_string", { cwd = false, word_match = "-w" }), desc = "Word(cwd)" },
    { "<leader>sW", Util.telescope("grep_string", { word_match = "-w" }), desc = "Word(root)" },
    { "<leader>sv", Util.telescope("grep_string", { cwd = false }), mode = "v", desc = "Selection(cwd)" },
    { "<leader>sV", Util.telescope("grep_string"), mode = "v", desc = "Selection(root)" },
    { "<leader>uC", Util.telescope("colorscheme", { enable_preview = true }), desc = "Colorscheme With Preview" },
    {
      "<leader>ss",
      Util.telescope("lsp_document_symbols", {
        symbols = {
          "Class",
          "Function",
          "Method",
          "Constructor",
          "Interface",
          "Module",
          "Struct",
          "Trait",
          "Field",
          "Property",
        },
      }),
      desc = "Symbols",
    },
    {
      "<leader>sS",
      Util.telescope("lsp_dynamic_workspace_symbols", {
        symbols = {
          "Class",
          "Function",
          "Method",
          "Constructor",
          "Interface",
          "Module",
          "Struct",
          "Trait",
          "Field",
          "Property",
        },
      }),
      desc = "Symbol(Workspace)",
    },
    {
      "<leader>fp",
      ":lua require'telescope'.extensions.project.project{display_type = 'full'}<CR>",
      desc = "Projects",
    },
  },
  opts = {
    defaults = {
      prompt_prefix = " ",
      selection_caret = "❆ ",
      mappings = {
        i = {
          -- This is `trouble` plugin's keymapping.
          ["<c-t>"] = function(...)
            return require("trouble.providers.telescope").open_with_trouble(...)
          end,
          ["<a-t>"] = function(...)
            return require("trouble.providers.telescope").open_selected_with_trouble(...)
          end,

          --
          ["<a-i>"] = function()
            local action_state = require("telescope.actions.state")
            local line = action_state.get_current_line()
            Util.telescope("find_files", { no_ignore = true, default_text = line })()
          end,
          ["<a-h>"] = function()
            local action_state = require("telescope.actions.state")
            local line = action_state.get_current_line()
            Util.telescope("find_files", { hidden = true, default_text = line })()
          end,

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
  },
  extensions = {
    project = project,
  },
}
