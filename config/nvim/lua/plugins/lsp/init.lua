-- Must use a init.lua since some files in "plugin/" not meet "LazyPluginSpec"
-- format, e.g. format.lua

-- Installed or reused.
local lsp_server = {
  clangd = {
    -- Set to false if you don't want this server to be installed by mason.
    mason = true,
    name = "mason-clangd",
    cmd = {
      "clangd",
      "-j=8",
      "--malloc-trim",
      "--background-index",
      "--pch-storage=memory",
    },
    filetypes = { "c", "cpp", "objc", "objcpp", "cuda", "proto", "cu", "su" },
  },

  lua_ls = {
    settings = {
      Lua = {
        workspace = {
          checkThirdParty = false,
        },
        completion = {
          callSnippet = "Replace",
        },
        -- To suppress undefined global vim
        diagnostics = {
          globals = { "vim" },
        },
      },
    },
  },

  pylsp = {
    settings = {
      pylsp = {
        plugins = {
          -- formatter options
          black = { enabled = false },    -- Too strict
          autopep8 = { enabled = false }, -- Too Old
          yapf = { enabled = true },      -- Enough

          -- linter options
          pylint = { enabled = true, executable = "pylint" },
          pyflakes = { enabled = false },
          pycodestyle = { enabled = false },

          -- type checker
          pylsp_mypy = { enabled = false },

          -- auto-completion options
          jedi_completion = { fuzzy = false },

          -- import sorting
          pyls_isort = { enabled = false },
        },
      },
    },
  },

}

return {
  -- https://github.com/neovim/nvim-lspconfig
  -- Some plugins are installed in lspconfig in the following order
  --    neodev
  --    format
  --    keymap
  --    mason
  --    mason-lspconfig
  --    nvim-lightbulb
  {
    "neovim/nvim-lspconfig",
    -- event = { "BufReadPre", "BufNewFile" },
    event = "LazyFile",

    ---@class PluginLspOpts
    opts = {
      servers = lsp_server,

      diagnostics = {             -- The options for vim.diagnostic.config()
        underline = true,
        update_in_insert = false, -- Update diagnostics in Insert mode.
        virtual_text = {
          prefix = "󰇥",
          spacing = 2,        -- Amount of empty spaces inserted at the beginning
          source = "if_many", -- Use "if_many" to only show sources if there is more than one diagnostic source in the buffer.
        },
        severity_sort = true,
        float = {
          border = "single",
        },
      },

      inlay_hints = {
        enabled = false,
      },

      autoformat = true,   -- Automatically format on save
      format = {           -- This option will be used in format.lua
        formatting_options = nil,
        timeout_ms = 2000, -- Time in milliseconds to block for formatting requests.
      },
    },

    ---@param opts PluginLspOpts which we configured above.
    config = function(_, opts)
      require("neodev").setup({})

      -- Setup formatting and keymaps.
      require("plugins.lsp.format").setup(opts)
      require("util").on_attach(function(client, buffer)
        require("plugins.lsp.keymaps").set_keymap(client, buffer)
      end)

      -- Update keymaps when capability changes.
      -- Set keymap in the initialization of lsp server and client.
      -- The client/registerCapability request is sent from the server to the
      -- client to register for a new capability on the client side. Not all
      -- clients need to support dynamic capability registration.
      -- This request is used during the initial communication between the lsp
      -- server and the client.
      local old_register_capability = vim.lsp.handlers["client/registerCapability"]
      vim.lsp.handlers["client/registerCapability"] = function(err, res, ctx)
        local ret = old_register_capability(err, res, ctx)
        local buffer = vim.api.nvim_get_current_buf()
        require("plugins.lsp.keymaps").set_keymap(_, buffer)
        return ret
      end

      -- Diagnostics icon
      for name, icon in pairs(require("config").icons.diagnostics) do
        name = "DiagnosticSign" .. name
        vim.fn.sign_define(name, { text = icon, texthl = name, numhl = "" })
      end

      -- Diagnostics
      vim.diagnostic.config(vim.deepcopy(opts.diagnostics))

      -- Inlay hint
      local inlay_hint = vim.lsp.buf.inlay_hint or vim.lsp.inlay_hint
      if opts.inlay_hints.enabled and inlay_hint then
        require("util").on_attach(function(client, buffer)
          if client.supports_method("textDocument/inlayHint") then
            inlay_hint(buffer, true)
          end
        end)
      end

      -- Virtual text
      if type(opts.diagnostics.virtual_text) == "table" and opts.diagnostics.virtual_text.prefix == "icons" then
        opts.diagnostics.virtual_text.prefix = function(diagnostic)
          local icons = require("config").icons.diagnostics
          for d, icon in pairs(icons) do
            if diagnostic.severity == vim.diagnostic.severity[d:upper()] then
              return icon
            end
          end
        end
      end

      -- Server and capabilities.
      local servers = opts.servers
      local capabilities = vim.tbl_deep_extend(
        "force",
        {},
        vim.lsp.protocol.make_client_capabilities(),
        require("cmp_nvim_lsp").default_capabilities() or {}
      )
      -- The client capabilities aren't be nil table.
      assert(type(capabilities) ~= "nil")

      -- Setup a specific server by lspconfig.
      local function setup(server)
        local server_opts = vim.tbl_deep_extend("force", {
          capabilities = vim.deepcopy(capabilities),
        }, servers[server] or {})

        require("lspconfig")[server].setup(server_opts)
      end


      -- The mason must be setup ahead of mason-lspconfig.
      -- The ensure installed server will be installed here.
      -- mason.nvim is optimized to load as little as possible during setup.
      -- Lazy-loading the plugin, or somehow deferring the setup, is not
      -- recommended.
      require("mason").setup()
      require("mason-lspconfig").setup({ handlers = { setup } })
      require("nvim-lightbulb").setup({
        autocmd = { enabled = true },
        sign = { text = "" },
      })
    end,
  },

  -- https://github.com/williamboman/mason.nvim
  -- A package manager.
  -- Setup this plugin at lspconfig.
  {
    "williamboman/mason.nvim",
    cmd = "Mason",
    keys = { { "<leader>cm", "<cmd>Mason<cr>", desc = "Mason" } },
    build = ":MasonUpdate",
  },

  -- https://github.com/williamboman/mason-lspconfig.nvim
  -- Extension to mason.nvim that makes it easier to use lspconfig with
  -- mason.nvim.
  -- Setup this plugin at lspconfig.
  {
    "williamboman/mason-lspconfig.nvim",
  },

  -- https://github.com/folke/neodev.nvim
  -- Full signature help, docs and completion for the nvim lua API.
  -- Setup this plugin at lspconfig.
  {
    "folke/neodev.nvim",
  },

  -- https://github.com/jose-elias-alvarez/null-ls.nvim
  -- A lua language server protocol server.
  -- {
  --   "jose-elias-alvarez/null-ls.nvim",
  --   event = { "BufReadPre", "BufNewFile" },
  --   opts = function()
  --     local nls = require("null-ls")
  --     return {
  --       root_dir = require("null-ls.utils").root_pattern(".null-ls-root", ".neoconf.json", "Makefile", ".git"),
  --       sources = {
  --         nls.builtins.formatting.fish_indent,
  --         nls.builtins.diagnostics.fish,
  --         nls.builtins.formatting.stylua,
  --         nls.builtins.formatting.shfmt,
  --       },
  --     }
  --   end,
  -- },

  -- https://github.com/kosayoda/nvim-lightbulb
  -- shows a lightbulb in the sign column whenever a textDocument/codeAction is available at the current cursor position.
  -- We need this since since without this we can't get a hint whehter current line has a code action.
  {
    "kosayoda/nvim-lightbulb",
  },
}
