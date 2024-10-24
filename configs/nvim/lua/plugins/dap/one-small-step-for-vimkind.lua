-- https://github.com/jbyuki/one-small-step-for-vimkind
-- Debug adapter for Neovim plugins

return {
  "jbyuki/one-small-step-for-vimkind",
  -- stylua: ignore
  keys = {
    { "<leader>daL", function() require("osv").launch({ port = 8086 }) end, desc = "Adapter Lua Server" },
    { "<leader>dal", function() require("osv").run_this() end,              desc = "Adapter Lua" },
  },
  config = function()
    local dap = require("dap")
    dap.adapters.nlua = function(callback, cfg)
      ---@diagnostic disable-next-line: undefined-field
      callback({ type = "server", host = cfg.host or "127.0.0.1", port = cfg.port or 8086 })
    end
    dap.configurations.lua = {
      {
        type = "nlua",
        request = "attach",
        name = "Attach to running Neovim instance",
      },
    }
  end,
}
