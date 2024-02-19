-----------------------------
----------------------------- Lsp Keymaps
-----------------------------
local M = {}

---@type PluginLspKeys
M.keys = nil

-- Return a table contains all keys.
-- @return (LazyKeys|{has?:string})[]
function M.get()
  if not M.keys then
    -- stylua: ignore start
    ---@class PluginLspKeys
    M.keys = {
      { "<leader>cd", vim.diagnostic.open_float,                                                              desc = "Diagnostic" },
      { "<leader>cl", "<cmd>LspInfo<cr>",                                                                     desc = "Lsp info" },
      { "gd",         function() require("telescope.builtin").lsp_definitions({ reuse_win = true }) end,      desc = "Goto definition",        has = "definition", },
      { "gr",         "<cmd>Telescope lsp_references<cr>",                                                    desc = "Goto references" },
      { "gD",         vim.lsp.buf.declaration,                                                                desc = "Goto declaration" },
      { "gI",         function() require("telescope.builtin").lsp_implementations({ reuse_win = true }) end,  desc = "Goto implementation", },
      { "gy",         function() require("telescope.builtin").lsp_type_definitions({ reuse_win = true }) end, desc = "Goto T[y]pe Definition", },
      { "K",          vim.lsp.buf.hover,                                                                      desc = "Hover" },
      { "gK",         vim.lsp.buf.signature_help,                                                             desc = "Signature help",         has = "signatureHelp" },
      { "]d",         M.diagnostic_goto(true),                                                                desc = "Next diagnostic" },
      { "[d",         M.diagnostic_goto(false),                                                               desc = "Prev diagnostic" },
      { "]e",         M.diagnostic_goto(true, "ERROR"),                                                       desc = "Next error" },
      { "[e",         M.diagnostic_goto(false, "ERROR"),                                                      desc = "Prev error" },
      { "]w",         M.diagnostic_goto(true, "WARN"),                                                        desc = "Next warning" },
      { "[w",         M.diagnostic_goto(false, "WARN"),                                                       desc = "Prev warning" },
      { "<leader>cf", function() require("plugins.lsp.format").format({ force = true }) end,                  desc = "Format" },
      { "<leader>cr", vim.lsp.buf.rename,                                                                     desc = "Rename",                 has = "rename" },
      { "<leader>ca", vim.lsp.buf.code_action,                                                                desc = "Code action",            mode = { "n", "v" },  has = "codeAction" },
      -- stylua: ignore end
      {
        "<leader>cA",
        function()
          vim.lsp.buf.code_action({
            context = {
              only = {
                "source",
              },
              diagnostics = {},
            },
          })
        end,
        desc = "Source Action",
        has = "codeAction",
      },
    }
  end
  return M.keys
end

-- The shortcut key of a method is registered only when the lsp client supports
-- the method. This function is used to determine whether a specific lsp client
-- supports a certain method.
---@param method string
function M.support_method(buffer, method)
  -- Make sure method equals "xxx/${method}"
  method = method:find("/") and method or "textDocument/" .. method

  local clients = vim.lsp.get_clients({ bufnr = buffer })
  for _, client in ipairs(clients) do
    if client.supports_method(method) then
      return true
    end
  end
  return false
end

-- Get keymaps which could be used directly by vim api.
function M.resolve()
  local Keys = require("lazy.core.handler.keys")
  --@type table<string,LazyKeys|{has?:string}>
  local keymaps = {}

  local function add(keymap)
    local keys = Keys.parse(keymap)
    if keys[2] == false then
      keymaps[keys.id] = nil
    else
      keymaps[keys.id] = keys
    end
  end

  -- Add default keymaps.
  for _, keymap in ipairs(M.get()) do
    add(keymap)
  end
  return keymaps
end

-- Set keymaps.
function M.set_keymap(_, buffer)
  local Keys = require("lazy.core.handler.keys")
  local keymaps = M.resolve()

  for _, keys in pairs(keymaps) do
    if not keys.has or M.support_method(buffer, keys.has) then
      local opts = Keys.opts(keys)
      ---@diagnostic disable-next-line: inject-field
      opts.has = nil

      ---@diagnostic disable-next-line: inject-field
      opts.silent = opts.silent ~= false

      ---@diagnostic disable-next-line: inject-field
      opts.buffer = buffer
      vim.keymap.set(keys.mode or "n", keys.lhs, keys.rhs, opts)
    end
  end
end

-- If next == true, then goto next diagnostic. Otherwise goto previous diagnostic.
-- The serverity identifies diagnostic level.
function M.diagnostic_goto(next, severity)
  local go = next and vim.diagnostic.goto_next or vim.diagnostic.goto_prev
  severity = severity and vim.diagnostic.severity[severity] or nil
  return function()
    go({ severity = severity })
  end
end

return M
