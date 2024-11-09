


def get_appended_id(source: str, seperator='_'):
    try:
        sloc = source.rfind(seperator)
        return source[:sloc], source[sloc+1:]
    except Exception:
        return source, None
