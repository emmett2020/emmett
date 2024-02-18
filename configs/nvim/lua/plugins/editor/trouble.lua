-- https://github.com/folke/trouble.nvim
-- A pretty diagnostics, references, telescope results, quickfix and location
-- list to help you solve all the trouble your code is causing.

return {
  "folke/trouble.nvim",
  cmd = { "TroubleToggle", "Trouble" },
  opts = {
    -- enabling this will use the signs defined in your lsp client
    use_diagnostic_signs = true,

    -- automatically close the list when you have no diagnostics
    auto_close = true,

    signs = {
      -- icons / text used for a diagnostic
      error = "",
      warning = "",
      hint = "",
      information = "",
      other = "",
    },
  },
  keys = {
    { "<leader>xx", "<cmd>TroubleToggle document_diagnostics<cr>",  desc = "Document Diagnostics" },
    { "<leader>xX", "<cmd>TroubleToggle workspace_diagnostics<cr>", desc = "Workspace Diagnostics" },
    {
      "[q",
      function()
        local trouble = require("trouble")
        if not trouble.is_open() then
          trouble.toggle("document_diagnostics")
        end
        trouble.previous({ skip_groups = true, jump = true })
      end,
      desc = "Prev diagnostic",
    },
    {
      "]q",
      function()
        local trouble = require("trouble")
        if not trouble.is_open() then
          trouble.toggle("document_diagnostics")
        end
        trouble.next({ skip_groups = true, jump = true })
      end,
      desc = "Next diagnostic",
    },
  },
}
