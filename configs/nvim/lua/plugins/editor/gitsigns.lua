-- https://github.com/lewis6991/gitsigns.nvim
-- Git signs highlights text that has changed since the list git commit, and
-- also lets you interactively stage & unstage hunks in a commit.

return {
  "lewis6991/gitsigns.nvim",
  event = "LazyFile",
  opts = {
    signs = {
      add = { text = "│" },
      change = { text = "│" },
      delete = { text = "_" },
      topdelete = { text = "‾" },
      changedelete = { text = "~" },
      untracked = { text = "┆" },
    },
    on_attach = function(buffer)
      local gs = package.loaded.gitsigns

      local function map(mode, l, r, desc)
        vim.keymap.set(mode, l, r, { buffer = buffer, desc = desc })
      end

      -- stylua: ignore start
      map("n", "]h", gs.next_hunk, "Next hunk") -- hunk is a piece of modified/added/deleted code
      map("n", "[h", gs.prev_hunk, "Prev hunk")

      map("n", "<leader>gs", "<cmd>Gitsigns show<cr>", "Show origin file")
      map("n", "<leader>gb", function() gs.blame_line({ full = true }) end, "Blame line")
      map("n", "<leader>gd", gs.diffthis, "Diff this")
      map({ "o", "x" }, "ih", "<cmd>Gitsigns select_hunk<cr>", "GitSigns select hunk")

      map("n", "<leader>gos", "<cmd>Gitsigns toggle_signs<cr>", "Toggle sign")
      map("n", "<leader>gob", "<cmd>Gitsigns toggle_current_line_blame<cr>", "Toggle current line blame")
      map("n", "<leader>goh", "<cmd>Gitsigns toggle_linehl<cr>", "Toggle line highlights")
      map("n", "<leader>god", "<cmd>Gitsigns toggle_deleted<cr>", "Toggle deleted")
      map("n", "<leader>gon", "<cmd>Gitsigns toggle_numhl<cr>", "Toggle num highlights")
      map("n", "<leader>gow", "<cmd>Gitsigns toggle_word_diff<cr>", "Toggle word diff")

      -- hunk and buffers
      map("n", "<leader>ghp", gs.preview_hunk, "Preview Hunk")
      map({ "n", "v" }, "<leader>ghs", "<cmd>Gitsigns stage_hunk<cr>", "Stage hunk")
      map({ "n", "v" }, "<leader>ghr", "<cmd>Gitsigns reset_hunk<cr>", "Reset hunk")
      map("n", "<leader>ghS", gs.stage_buffer, "Stage buffer")
      map("n", "<leader>ghR", gs.reset_buffer, "Reset Buffer")
      -- stylua: ignore end
    end,
  },
}
