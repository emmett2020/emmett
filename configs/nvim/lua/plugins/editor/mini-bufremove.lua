-- https://github.com/echasnovski/mini.bufremove
-- This plugin is used to delete the current buffer. Compared to calling native
-- functions directly, this plug-in encapsulates some necessary functions. For
-- example, it prompts you whether to force a buffer with unsaved changes to be
-- deleted, and to select the next buffer that should be displayed after a
-- buffer is deleted.
return {
  "echasnovski/mini.bufremove",
  keys = {
    {
      "<leader>bd",
      function()
        -- "0" for current,
        -- "false" for not force delete.
        require("mini.bufremove").delete(0, false)
      end,
      desc = "Delete current buffer",
    },
    {
      "<leader>bD",
      function()
        -- "0" for current,
        -- "false" for not force delete.
        require("mini.bufremove").delete(0, true)
      end,
      desc = "Delete current buffer(Force)",
    },
  },
}
