-------------------------------------------------------------
---                      Keymaps                          ---
-------------------------------------------------------------
-- This file is automatically loaded by config.init
-- You may find keymaps at there: https://neovim.io/doc/user/intro.html#notation

local Util = require("util")

local function map_shortcut(mode, lhs, rhs, opts)
  local keys = require("lazy.core.handler").handlers.keys
  ---@cast keys LazyKeysHandler
  -- https://github.com/folke/lazy.nvim/blob/main/lua/lazy/core/handler/keys.lua
  -- Do not create the keymap if a LazyKeysHandler exists.
  if not keys.active[keys.parse({ lhs, mode = mode, id = "" }).id] then
    opts = opts or {}
    opts.silent = opts.silent ~= false
    if opts.remap and not vim.g.vscode then
      opts.remap = nil
    end
    vim.keymap.set(mode, lhs, rhs, opts)
  end
end

------------------------
------------------------ Window
------------------------
-- Split/Create window.
map_shortcut("n", "<leader>w-", "<C-W>s", { desc = "Split window below", remap = true })
map_shortcut("n", "<leader>w|", "<C-W>v", { desc = "Split window right", remap = true })
map_shortcut("n", "<leader>-", "<C-W>s", { desc = "Split window below", remap = true })
map_shortcut("n", "<leader>|", "<C-W>v", { desc = "Split window right", remap = true })

-- Delete window.
map_shortcut("n", "<leader>wd", "<C-W>c", { desc = "Delete window", remap = true })

-- Go to window.
map_shortcut("n", "<C-h>", "<C-w>h", { desc = "Left window", remap = true })
map_shortcut("n", "<C-j>", "<C-w>j", { desc = "Lower window", remap = true })
map_shortcut("n", "<C-k>", "<C-w>k", { desc = "Upper window", remap = true })
map_shortcut("n", "<C-l>", "<C-w>l", { desc = "Right window", remap = true })
map_shortcut("n", "<leader>wh", "<C-w>h", { desc = "Left window", remap = true })
map_shortcut("n", "<leader>wj", "<C-w>j", { desc = "Lower window", remap = true })
map_shortcut("n", "<leader>wk", "<C-w>k", { desc = "Upper window", remap = true })
map_shortcut("n", "<leader>wl", "<C-w>l", { desc = "Right window", remap = true })
map_shortcut("n", "<leader>ww", "<C-W>p", { desc = "Other window", remap = true })

-- Resize window.
-- When using the leader w to adjust the window size, you need to press many keys, so it is only used for fine adjustment.
-- When using ctrl to adjust the window size, you can hold down the ctrl key and not let go, so the adjustment is quick and used for large adjustments.
map_shortcut("n", "<C-Up>", "<cmd>resize +4<cr>", { desc = "Increase window height" })
map_shortcut("n", "<C-Down>", "<cmd>resize -4<cr>", { desc = "Decrease window height" })
map_shortcut("n", "<C-Left>", "<cmd>vertical resize +4<cr>", { desc = "Increase window width" })
map_shortcut("n", "<C-Right>", "<cmd>vertical resize -4<cr>", { desc = "Decrease window width" })
map_shortcut("n", "<leader>w<Up>", "<cmd>resize +2<cr>", { desc = "Increase window height" })
map_shortcut("n", "<leader>w<Down>", "<cmd>resize -2<cr>", { desc = "Decrease window height" })
map_shortcut("n", "<leader>w<Left>", "<cmd>vertical resize +2<cr>", { desc = "Increase window width" })
map_shortcut("n", "<leader>w<Right>", "<cmd>vertical resize -2<cr>", { desc = "Decrease window width" })

------------------------
------------------------ Lines
------------------------
-- Move Lines.
map_shortcut("n", "<A-j>", "<cmd>m .+1<cr>==", { desc = "Move down" })
map_shortcut("n", "<A-k>", "<cmd>m .-2<cr>==", { desc = "Move up" })
map_shortcut("i", "<A-j>", "<esc><cmd>m .+1<cr>==gi", { desc = "Move down" })
map_shortcut("i", "<A-k>", "<esc><cmd>m .-2<cr>==gi", { desc = "Move up" })
map_shortcut("v", "<A-j>", ":m '>+1<cr>gv=gv", { desc = "Move down" })
map_shortcut("v", "<A-k>", ":m '<-2<cr>gv=gv", { desc = "Move up" })

------------------------
------------------------ Buffer
------------------------
-- Delete buffer use miniremove plugin.
-- Go to buffer.
map_shortcut("n", "<leader>bb", "<cmd>e #<cr>", { desc = "Other buffer" })
if Util.has("bufferline.nvim") then
  map_shortcut("n", "[b", "<cmd>BufferLineCyclePrev<cr>", { desc = "Prev buffer" })
  map_shortcut("n", "]b", "<cmd>BufferLineCycleNext<cr>", { desc = "Next buffer" })
  map_shortcut("n", "<leader>bh", "<cmd>BufferLineCyclePrev<cr>", { desc = "Prev buffer" })
  map_shortcut("n", "<leader>bj", "<cmd>BufferLineCycleNext<cr>", { desc = "Next buffer" })
else
  map_shortcut("n", "[b", "<cmd>bprevious<cr>", { desc = "Prev buffer" })
  map_shortcut("n", "]b", "<cmd>bnext<cr>", { desc = "Next buffer" })
  map_shortcut("n", "<leader>bh", "<cmd>bprevious<cr>", { desc = "Prev buffer" })
  map_shortcut("n", "<leader>bl", "<cmd>bnext<cr>", { desc = "Next buffer" })
end

-- Save buffer.
map_shortcut({ "i", "v", "n", "s" }, "<C-s>", "<cmd>w<cr><esc>", { desc = "Save buffer" })
map_shortcut({ "v", "n" }, "<leader>.", "<cmd>w<cr><esc>", { desc = "Save buffer" })

-- Create buffer.
map_shortcut("n", "<leader>bn", "<cmd>enew<cr>", { desc = "New buffer" })

------------------------
------------------------ Search
------------------------

-- Go to search result by '*' or '/' command.
-- https://github.com/mhinz/vim-galore#saner-behavior-of-n-and-n
map_shortcut("n", "n", "'Nn'[v:searchforward]", { expr = true, desc = "Next search result" })
map_shortcut("x", "n", "'Nn'[v:searchforward]", { expr = true, desc = "Next search result" })
map_shortcut("o", "n", "'Nn'[v:searchforward]", { expr = true, desc = "Next search result" })
map_shortcut("n", "N", "'nN'[v:searchforward]", { expr = true, desc = "Prev search result" })
map_shortcut("x", "N", "'nN'[v:searchforward]", { expr = true, desc = "Prev search result" })
map_shortcut("o", "N", "'nN'[v:searchforward]", { expr = true, desc = "Prev search result" })

-- Go to search result by 'f' command.
-- Add undo break-points.
map_shortcut("i", ",", ",<c-g>u")
map_shortcut("i", ".", ".<c-g>u")
map_shortcut("i", ";", ";<c-g>u")

-- Clear search result with <esc>
map_shortcut({ "i", "n" }, "<esc>", "<cmd>noh<cr><esc>", { desc = "Escape and clear hlsearch" })

-- Clear search, diff update and redraw result.
-- taken from runtime/lua/_editor.lua
map_shortcut(
  "n",
  "<leader>ur",
  "<Cmd>nohlsearch<Bar>diffupdate<Bar>normal! <C-L><CR>",
  { desc = "Redraw / clear hlsearch / diff update" }
)

------------------------
------------------------ Diagnostics and quickfix
------------------------
map_shortcut("n", "<leader>xl", "<cmd>lopen<cr>", { desc = "Location List" })
map_shortcut("n", "<leader>xq", "<cmd>copen<cr>", { desc = "Quickfix List" })
if not Util.has("trouble.nvim") then
  map_shortcut("n", "[q", vim.cmd.cprev, { desc = "Prev quickfix" })
  map_shortcut("n", "]q", vim.cmd.cnext, { desc = "Next quickfix" })
end

------------------------
------------------------ Toggle options
------------------------
map_shortcut("n", "<leader>of", require("plugins.lsp.format").toggle, { desc = "Toggle format on Save" })
map_shortcut("n", "<leader>os", function()
  Util.toggle("spell")
end, { desc = "Toggle Spelling" })
map_shortcut("n", "<leader>ow", function()
  Util.toggle("wrap")
end, { desc = "Toggle Word Wrap" })
map_shortcut("n", "<leader>ol", function()
  Util.toggle_number()
end, { desc = "Toggle Line Numbers" })
map_shortcut("n", "<leader>od", Util.toggle_diagnostics, { desc = "Toggle Diagnostics" })
local conceallevel = vim.o.conceallevel > 0 and vim.o.conceallevel or 3
map_shortcut("n", "<leader>oc", function()
  Util.toggle("conceallevel", false, { 0, conceallevel })
end, { desc = "Toggle Conceal" })
if vim.lsp.inlay_hint then
  map_shortcut("n", "<leader>oh", function()
    vim.lsp.inlay_hint(0, nil)
  end, { desc = "Toggle Inlay Hints" })
end

------------------------
------------------------ Terminal
------------------------
-- Go to terminal
map_shortcut("t", "<C-h>", "<cmd>wincmd h<cr>", { desc = "Left window" })
map_shortcut("t", "<C-j>", "<cmd>wincmd j<cr>", { desc = "Lower window" })
map_shortcut("t", "<C-k>", "<cmd>wincmd k<cr>", { desc = "Upper window" })
map_shortcut("t", "<C-l>", "<cmd>wincmd l<cr>", { desc = "Right window" })

-- Hide or show terminal
map_shortcut("t", "<C-/>", "<cmd>close<cr>", { desc = "Hide/Show Terminal" })

-- Control terminal.
map_shortcut("t", "<c-_>", "<cmd>close<cr>", { desc = "which_key_ignore" })
map_shortcut("t", "<esc>", "<c-\\><c-n>", { desc = "Enter Normal Mode" })

-- Floating terminal.
-- Change this to "/bin/zsh" or others if you need.
local terminal_shell_cmd = "/bin/bash"
local lazyterm = function()
  Util.float_term(terminal_shell_cmd, { cwd = Util.get_root() })
end
map_shortcut("n", "<leader>fT", lazyterm, { desc = "Terminal(root)" })
map_shortcut("n", "<leader>ft", function()
  Util.float_term(terminal_shell_cmd)
end, { desc = "Terminal(cwd)" })
map_shortcut("n", "<c-/>", lazyterm, { desc = "Terminal(root)" })
map_shortcut("n", "<c-_>", lazyterm, { desc = "which_key_ignore" })

------------------------
------------------------ Utilities
------------------------

-- Keywordprg.
map_shortcut("n", "<leader>K", "<cmd>norm! K<cr>", { desc = "Keywordprg" })

-- Better indenting.
map_shortcut("v", "<", "<gv")
map_shortcut("v", ">", ">gv")

-- Lazy nvim.
map_shortcut("n", "<leader>l", "<cmd>Lazy<cr>", { desc = "Lazy" })

-- Lazygit
map_shortcut("n", "<leader>gG", function()
  Util.float_term({ "lazygit" }, { cwd = Util.get_root(), esc_esc = false, ctrl_hjkl = false })
end, { desc = "Lazygit(root)" })
map_shortcut("n", "<leader>gg", function()
  Util.float_term({ "lazygit" }, { esc_esc = false, ctrl_hjkl = false })
end, { desc = "Lazygit(cwd)" })

-- quit
map_shortcut("n", "<leader>qq", "<cmd>qa<cr>", { desc = "Quit nvim" })

-- Better move.
map_shortcut({ "n", "x" }, "j", "v:count == 0 ? 'gj' : 'j'", { expr = true, silent = true })
map_shortcut({ "n", "x" }, "k", "v:count == 0 ? 'gk' : 'k'", { expr = true, silent = true })
