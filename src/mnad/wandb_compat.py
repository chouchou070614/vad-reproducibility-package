try:
    import wandb as wandb
except ImportError:
    class _DisabledWandb:
        def init(self, *args, **kwargs):
            return self

        def watch(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

        def finish(self, *args, **kwargs):
            return None

    wandb = _DisabledWandb()
