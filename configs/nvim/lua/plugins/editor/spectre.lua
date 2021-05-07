-- https://github.com/nvim-pack/nvim-spectre
-- search/replace in multiple files
return {
  "nvim-pack/nvim-spectre",
  cmd = "Spectre",
  keys = {
    -- stylua: ignore start
    { "<leader>sr", function() require("spectre").open() end,             desc = "Replace" },
    { "<leader>sR", function() require('spectre').open_file_search() end, desc = "Replace in current buffer" },
    -- stylua: ignore end
  },
  opts = {
    color_devicons = true,
    open_cmd = "noswapfile vnew",
  },
}
