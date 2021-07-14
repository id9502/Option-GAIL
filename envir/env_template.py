class EnvTemplate(object):
    def __init__(self, task_name: str = "HalfCheetah-v2", *args, **kwgs):
        self._display = False
        # store const value but do not instantiate
        pass

    def init(self, display=False):
        self._display = display
        # do all instantiation here
        return self

    def reset(self, random: bool = False):
        # do your env reset here and get real s
        s = None
        # ----

        return s

    def step(self, a):
        # do your env step here and get real (s, r, t)
        s, r, t = None, None, None
        # -----

        if self._display:
            self.render()
        return s, r, t

    def render(self):
        pass

    def state_action_size(self):
        # this will be called when algorithm inits, sometimes before self.init() called,
        # so do not depends on self.init()

        # return real s_dim, a_dim
        s_dim, a_dim = None, None
        # ----

        return s_dim, a_dim
