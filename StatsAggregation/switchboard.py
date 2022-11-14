class SwitchBoard:
    sinks = dict()
    sources = dict()
    switchboard = dict()

    @classmethod
    def register_source(cls, name):
        def _inner(f):
            cls.sources[name] = f
            f._sname = name
            return f
        return _inner

    @classmethod
    def request_source(cls, name, sink_name=None):
        def _inner(f):
            sname = f.__name__ if sink_name is None else sink_name
            f.sname = sname
            cls.sinks[sname] = f
            cls.switchboard[sname] = name
            return f
        return _inner

    @classmethod
    def get_source(cls, sink_func):
        try:
            source_func_name = cls.switchboard[sink_func.sname]
        except AttributeError:
            raise SwitchBoardError("sink function not registered")
        try:
            source_func = cls.sources[source_func_name]
        except KeyError:
            raise SwitchBoardError("source function not registered")
        return source_func


class SwitchBoardError(Exception):
    pass
