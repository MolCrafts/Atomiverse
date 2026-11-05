# Atomiverse history rewrite

`origin/dev` was rewritten in place. The old tip and the new tip have the
same tree and different commit ids.

| | Commit |
| --- | --- |
| Old `dev` tip | `a14449d8` (`a14449d81fd9d958bab71092183f9dd3c9f49677`) |
| New `origin/dev` | `bbff5f06` (`bbff5f06d652634cdc3ce02df890ccefae82d6b6`) |

Run a command once, from a checkout that still contains the old commit named
in it. After `dev` already descends from `bbff5f06`, do not run the `dev`
command again. Do not `git push --force` or `--force-with-lease` to
`origin` or `upstream`, on any branch. A rejected non-fast-forward push is
the rewrite doing its job. Leave it rejected.

## dev

Commits made on top of the old tip, and a dirty worktree, move with:

```bash
git fetch origin
git rebase --autostash --onto origin/dev a14449d8 dev
```

That replays only commits after `a14449d8`. It does not push.

## Other local branches

`8429ea47` and `543ebd8f` were not rewritten.

```bash
git fetch origin

# tip a3bc83ec is now f53ca84c; no unique commits
git rebase --autostash --onto f53ca84c7bb69f53d1cc1833d2cd3eb5ff02e490 a3bc83ec73c2ff6f838d1445b0a8750e40ef1a19 chore/clang-format-onto-dev

# one commit on top of old ee2315ca, which is now 199ec5be
git rebase --autostash --onto 199ec5be09c7fb5f3133a7e8e48e7c2e04d39e97 ee2315cad61416ea697024da9dfaff42ffa3ac96 chore/single-branch-master

# tip b204d871 is now c3b1930f; no unique commits
git rebase --autostash --onto c3b1930f6a0443875a2bf2e7b5e93de3f1698777 b204d8716dd0a16bec24a87bd597e02595f9eea3 feat/comm-topology

# tip 272a7166 is now 918a7ec2; no unique commits
git rebase --autostash --onto 918a7ec251a04f998b83058e4f3a2ec2244500b2 272a7166b7f72c379178c0dc69da3ad9fcac83bc feat/dftb-data-00-honesty
```

`archive/impl-kokkos-wip` forks at `8429ea47`, which still exists on the new
history. Do not rebase it.

`master` is still `543ebd8f`. Do not rebase it onto `origin/dev`. Catching
up with `origin/master` is a fast-forward only:

```bash
git switch master
git merge --ff-only origin/master
git switch dev
```

None of these commands publish anything.
