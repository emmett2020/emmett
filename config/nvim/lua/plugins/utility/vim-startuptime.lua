-- https://github.com/dstein64/vim-startuptime
return {
  "dstein64/vim-startuptime",
  cmd = "StartupTime",
  config = function()
    vim.g.startuptime_tries = 10
  end,

}
