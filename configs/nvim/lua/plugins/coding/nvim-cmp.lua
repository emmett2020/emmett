-- https://github.com/hrsh7th/nvim-cmp
-- Wiki: https://github.com/hrsh7th/nvim-cmp/wiki
-- nvim-cmp: Automatic completion.

-- Currently used sources:
-- 1.lsp        https://github.com/hrsh7th/cmp-nvim-lsp
-- 2.buffer     https://github.com/hrsh7th/cmp-buffer
-- 3.path       https://github.com/hrsh7th/cmp-path
-- 4.lua        https://github.com/saadparwaiz1/cmp_luasnip
--              https://github.com/L3MON4D3/LuaSnip
-- More sources will be found at:
--              https://github.com/hrsh7th/nvim-cmp/wiki/List-of-sources

return {
  "hrsh7th/nvim-cmp",

  -- 1.cmp-nvim-lsp: Serve more capabilities than default lsp client.
  -- 2.cmp-buffer:  have everything nicely collected in a single completion popup.
  -- 3.cmp-path:    paths of files and folders.
  -- 4.cmp_luasnip: luasnip completion source for nvim-cmp.
  dependencies = {
    "hrsh7th/cmp-nvim-lsp",
    "hrsh7th/cmp-buffer",
    "hrsh7th/cmp-path",
    "saadparwaiz1/cmp_luasnip",
  },

  -- Suggested usage by nvim-cmp.
  event = { "InsertEnter" },

  -- Real options.
  opts = function()
    local cmp = require("cmp")
    local defaults = require("cmp.config.default")()
    local luasnip = require("luasnip")

    local has_words_before = function()
      unpack = unpack or table.unpack
      local line, col = unpack(vim.api.nvim_win_get_cursor(0))
      return col ~= 0 and vim.api.nvim_buf_get_lines(0, line - 1, line, true)[1]:sub(col, col):match("%s") == nil
    end

    vim.api.nvim_set_hl(0, "CmpGhostText", { link = "Comment", default = true })

    return {
      completion = {
        completeopt = "menu,menuone,noinsert",
      },

      -- See the linakge above.
      sources = cmp.config.sources({
        { name = "nvim_lsp" },
        { name = "buffer" },
        { name = "path" },
        { name = "luasnip" },
      }),

      -- Enable luasnip.
      -- The detailed configuration of luasnip will be found at luanip.lua
      snippet = {
        expand = function(args)
          luasnip.lsp_expand(args.body)
        end,
      },

      -- Set keys. Only uses these three keys is enough.
      mapping = cmp.mapping.preset.insert({
        -- Use <Tab> to navigate between hint table.
        ["<Tab>"] = cmp.mapping(function(fallback)
          if cmp.visible() then
            cmp.select_next_item()
          elseif luasnip.expand_or_locally_jumpable() then
            luasnip.expand_or_jump()
          elseif has_words_before() then
            cmp.complete()
          else
            fallback()
          end
        end, { "i", "s" }),

        -- Backward.
        ["<S-Tab>"] = cmp.mapping(function(fallback)
          if cmp.visible() then
            cmp.select_prev_item()
          elseif luasnip.jumpable(-1) then
            luasnip.jump(-1)
          else
            fallback()
          end
        end, { "i", "s" }),

        -- Accept currently selected item.
        -- Set `select` to `false` to only confirm explicitly selected items.
        ["<CR>"] = cmp.mapping.confirm({ select = true }),
      }),

      -- Refine the apperance of nvim-cmp.
      formatting = {
        -- Icons
        format = function(_, item)
          local icons = require("config").icons.kinds
          if icons[item.kind] then
            item.kind = icons[item.kind] .. item.kind
          end
          return item
        end,
      },

      experimental = {
        ghost_text = {
          hl_group = "CmpGhostText",
        },
      },
      sorting = defaults.sorting,
    }
  end,
}
