-----------------------------
----------------------------- lsp format wrapper
-----------------------------
local Util = require("lazy.core.util")
local M = {}

---@class PluginLspOpts
M.opts = nil

---@param opts PluginLspOpts
function M.setup(opts)
  M.opts = opts
  vim.api.nvim_create_autocmd("BufWritePre", {
    group = vim.api.nvim_create_augroup("DailyFormat", {}),
    callback = function()
      if M.opts.autoformat then
        M.format()
      end
    end,
  })
end

-- Return whether enables auto format.
function M.enabled()
  return M.opts.autoformat
end

-- Toggle autoformat option.
function M.toggle()
  if vim.b.autoformat == false then
    vim.b.autoformat = nil
    M.opts.autoformat = true
  else
    M.opts.autoformat = not M.opts.autoformat
  end
  if M.opts.autoformat then
    Util.info("Enabled format on save", { title = "Format" })
  else
    Util.warn("Disabled format on save", { title = "Format" })
  end
end

-- Format a file.
---@param opts? {force?:boolean}
function M.format(opts)
  local buf = vim.api.nvim_get_current_buf()
  if vim.b.autoformat == false and not (opts and opts.force) then
    return
  end

  local clients = M.get_clients(buf)
  local client_ids = vim.tbl_map(function(client)
    return client.id
  end, clients)

  -- No active formatters for current buffer.
  if #client_ids == 0 then
    return
  end

  -- Notify formatting operation.
  -- Use noice.nvim to put notification in "mini" view.
  M.notify(clients)

  -- Real formatting operation.
  -- How does vim.lsp.buf.format() deal with multiple attached LSP clients?
  -- In default, all lsp client will be used in turn.
  -- We can use a filter function to always choose first client.
  vim.lsp.buf.format(vim.tbl_deep_extend("force", {
    bufnr = buf,
    filter = function(client)
      return vim.tbl_contains(client_ids, client.id)
    end,
  }, require("util").opts_of_plugin("nvim-lspconfig").format or {}))
end

-- Notify formatting operation.
---@param clients lsp.Client[]
function M.notify(clients)
  if #clients > 0 then
    local content = "Formatting this file uses " .. clients[1].name
    vim.notify(content, vim.log.levels.INFO)
  else
    local content = "Formatting failed. No active lsp client."
    vim.notify(content, vim.log.levels.ERROR)
  end
end

-- Gets all lsp clients that support formatting.
---@return lsp.Client[]
function M.get_clients(bufnr)
  -- Current active lsp client.
  local ret = {}

  ---@type lsp.Client[]
  local clients = vim.lsp.get_clients({ bufnr = bufnr })
  for _, client in ipairs(clients) do
    if M.supports_format(client) then
      table.insert(ret, client)
    end
  end

  return ret
end

-- Check whether a lsp client supports formatting and have not disabled it in
-- their client config.
---@param client lsp.Client
function M.supports_format(client)
  if
      client.config
      and client.config.capabilities
      ---@diagnostic disable-next-line: undefined-field
      and client.config.capabilities.documentFormattingProvider == false
  then
    -- Not support formatting.
    return false
  end
  return client.supports_method("textDocument/formatting") or client.supports_method("textDocument/rangeFormatting")
end

return M
