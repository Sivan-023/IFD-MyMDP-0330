# encoding=utf-8
def ClassifyEnv(mode, trainx, trainy):
    from .CMDP_Env import ClassifyEnv
    env = ClassifyEnv(mode, trainx, trainy)
    return env