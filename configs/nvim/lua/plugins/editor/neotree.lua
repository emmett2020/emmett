-- file explorer
-- https://github.com/nvim-neo-tree/neo-tree.nvim
-- Press ? in the Neo-tree window to view the list of mappings.
local Util = require("util")

return {
  "nvim-neo-tree/neo-tree.nvim",
  branch = "v3.x",
  cmd = "Neotree",
  keys = {
    {
      "<leader>fe",
      function()
        require("neo-tree.command").execute({ toggle = true, dir = Util.get_root() })
      end,
      desc = "File Explorer(root)",
    },
    {
      "<leader>fE",
      function()
        require("neo-tree.command").execute({ toggle = true, dir = vim.loop.cwd() })
      end,
      desc = "File Explorer(cwd)",
    },
    { "<leader>e", "<leader>fe", desc = "File Explorer(root)", remap = true },
    { "<leader>E", "<leader>fE", desc = "File Explorer(cwd)", remap = true },
  },
  deactivate = function()
    vim.cmd([[Neotree close]])
  end,
  init = function()
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
    sources = { "filesystem", "buffers", "git_status", "document_symbols" },

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

    buffers = {
      follow_current_file = {
        enabled = true, -- This will find and focus the file in the active buffer every time
        -- the current file is changed while the tree is open.
        leave_dirs_open = false, -- `false` closes auto expanded dirs, such as with `:Neotree reveal`
      },

      group_empty_dirs = false, -- when true, empty folders will be grouped together
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
  config = function(_, opts)
    require("neo-tree").setup(opts)
    vim.api.nvim_create_autocmd("TermClose", {
      pattern = "*lazygit",
      callback = function()
        if package.loaded["neo-tree.sources.git_status"] then
          require("neo-tree.sources.git_status").refresh()
        end
      end,
    })
  end,
}
