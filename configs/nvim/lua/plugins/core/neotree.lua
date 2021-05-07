-- https://github.com/nvim-neo-tree/neo-tree.nvim
-- file explorer
-- Press ? in the Neo-tree window to view the list of mappings.

return {
  "nvim-neo-tree/neo-tree.nvim",
  cmd = "Neotree",
  keys = {
    {
      "<leader>e",
      function()
        require("neo-tree.command").execute({ toggle = true, dir = vim.loop.cwd() })
      end,
      desc = "File explorer",
    },
    { "<leader>fel", "<cmd>Neotree filesystem reveal left<cr>", desc = "Move neotree to left" },
    { "<leader>fef", "<cmd>Neotree filesystem reveal float<cr>", desc = "Float neotree" },
  },

  deactivate = function()
    vim.cmd([[Neotree close]])
  end,

  init = function()
    -- If `nvim dir`, open a buffer to show items in this directory.
    if vim.fn.argc() == 1 then
      local item = vim.inspect(vim.fn.argv(0))
      local stat = vim.loop.fs_stat(item)
      if stat and stat.type == "directory" then
        require("neo-tree")
      end
    end
  end,

  opts = {
    popup_border_style = "rounded",
    enable_git_status = true,
    enable_diagnostics = true,
    sources = { "filesystem" },

    -- when opening files, do not use windows containing these filetypes or buftypes
    open_files_do_not_replace_types = { "terminal", "Trouble", "qf", "Outline" },

    filesystem = {
      filtered_items = {
        visible = true,
        hide_dotfiles = false,
        hide_gitignored = false,
      },
      bind_to_cwd = false,
      follow_current_file = { enabled = true },
      use_libuv_file_watcher = true,
    },

    window = {
      position = "float",
      mappings = {
        ["<space>"] = "none",
      },
    },

    default_component_configs = {
      indent = {
        with_expanders = true, -- if nil and file nesting is enabled, will enable expanders
        expander_collapsed = "",
        expander_expanded = "",
        expander_highlight = "NeoTreeExpander",
      },
    },
  },
}
