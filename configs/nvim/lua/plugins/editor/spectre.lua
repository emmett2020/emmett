-- search/replace in multiple files
-- https://github.com/nvim-pack/nvim-spectre
return {
  "nvim-pack/nvim-spectre",
  cmd = "Spectre",
    -- stylua: ignore
    keys = {
      { "<leader>sr", function() require("spectre").open() end, desc = "Replace" },
      { "<leader>sR", function() require('spectre').open_file_search()  end, desc = "Replace in current buffer" },
    },
  opts = {
    color_devicons = true,
    open_cmd = "noswapfile vnew",
  },
}
