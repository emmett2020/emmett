# Changelog
## 1.0.1 (2024-02-27)
### Bug Fixes
1. The nvim-cmp plugin supports jumping through snippets of code using the <tab> shortcut. However, when there are multiple jump points, using <tab> elsewhere in the code returns you to the next unjump point. This makes it impossible to predict the position of the cursor when the <tab> key is used. Remove this feature.

## 1.0.0 (2024-02-19)
### Bug Fixes
1. Fix too slow startup time when opening cpp files.

### Features
1. Add LazyFile event referenced to LazyVim. Thanks to @folke.
2. Organize all the plugins.
