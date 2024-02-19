-- https://github.com/akinsho/toggleterm.nvim
-- Terminal support

-- The toggleterm uses index as arugment to create new terminal.
local term_idx = 0 -- more than 999 for special command, e.g. lazygit

local function create_float_term()
  term_idx = term_idx + 1
  require("toggleterm").toggle(term_idx, 0, "", "float")
end

local function create_horizontal_term()
  term_idx = term_idx + 1
  require("toggleterm").toggle(term_idx, 15, "", "horizontal")
end

local function create_vertical_term()
  term_idx = term_idx + 1
  require("toggleterm").toggle(term_idx, vim.o.columns * 0.4, "", "vertical")
end

-- Lazygit
local created_lazygit = false
---@class toggleterm.terminal.Terminal
local lazygit = nil
local function Lazygit()
  if not created_lazygit then
    local Terminal = require("toggleterm.terminal").Terminal
    lazygit = Terminal:new({
      cmd = "lazygit",
      direction = "float",
      float_opts = { border = "double" },
      count = 1000, -- lazygit's index is 1000

      -- function to run on opening the terminal
      on_open = function(term)
        vim.cmd("startinsert!")
        vim.api.nvim_buf_set_keymap(term.bufnr, "n", "q", "<cmd>close<CR>", { noremap = true, silent = true })
        vim.keymap.set("t", "<esc>", "<esc>", { buffer = term.bufnr, nowait = true })
      end,

      -- function to run on closing the terminal
      on_close = function(_)
        vim.cmd("startinsert!")
      end,
    })

    created_lazygit = true
    lazygit:toggle()
  else
    ---@diagnostic disable-next-line: undefined-field
    lazygit:toggle()
  end
end

-- "clear" cann't clear scrollback, use this workaround.
local term_clear = function()
  vim.fn.feedkeys("^L", 'n')
  local sb = vim.bo.scrollback
  vim.bo.scrollback = 1
  vim.bo.scrollback = sb
end


return {
  "akinsho/toggleterm.nvim",
  version = "*",

  opts = {
    float_opts = {
      border = "single",
      width = function()
        return math.floor(vim.o.columns * 0.85)
      end,
      height = function()
        return math.floor(vim.o.lines * 0.85)
      end,
    }
  },
  config = function(_, opts)
    require("toggleterm").setup(opts)
  end,
  keys = {
    { "<leader>tf", create_float_term,      desc = "Float terminal" },
    { "<leader>th", create_horizontal_term, desc = "Horizontal terminal" },
    { "<leader>tv", create_vertical_term,   desc = "Vertical terminal" },

    {
      "<leader>tt",
      function()
        if term_idx == 0 then
          create_horizontal_term()
        else
          require("toggleterm").toggle_all()
        end
      end,
      desc = "Show/hidden all"
    },
    { "<leader>st", "<cmd>TermSelect <cr>", desc = "Terminal" },
    { "<leader>gg", Lazygit,                desc = "Lazygit" },
    { "<C-l>",      term_clear,             { mode = { 't', 'n' } } },
  }
}
