1. Always write shell scripts use bash grammar.
2. When you're running script in bash:
```bash
  # Run script in a new shell environment.
  ./test.sh

  # Run script in current shell.
  source ./test.sh
```
3. When you're running script in zsh:
```bash
  # Run script in a new shell environment.
  ./test.sh

  # Run script in current shell.
  # WARN: Don't do this. zsh is incompatible with bash especially combined with "source" command.
  # source ./test.sh
  eval "$(./test.sh)"
```

