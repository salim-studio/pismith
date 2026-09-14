"""CLI: python -m pismith profile data.csv | python -m pismith query sales.db "SELECT ..." """
import sys


def _cmd_profile(path: str):
    from .io import load
    from .data import profile
    import json
    rows = load(path)
    print(json.dumps(profile(rows), ensure_ascii=False, indent=2, default=str))


def _cmd_query(dbpath: str, sql: str):
    from .db import Database
    import json
    db = Database(dbpath)
    print(json.dumps(db.query(sql), ensure_ascii=False, indent=2, default=str)[:4000])
    db.close()


def main(argv=None):
    argv = list(argv or sys.argv[1:])
    if not argv or argv[0] in ("-h", "--help", "help"):
        print("pismith CLI — profile <file> | query <db> <sql> | demo | version")
        return 0
    cmd = argv[0]
    if cmd == "version":
        from . import __version__
        print(__version__)
    elif cmd == "profile" and len(argv) > 1:
        _cmd_profile(argv[1])
    elif cmd == "query" and len(argv) > 2:
        _cmd_query(argv[1], argv[2])
    elif cmd == "demo":
        from examples.examples_db_ml import main as demo
        demo()
    else:
        print(f"Unknown command: {cmd} — try: python -m pismith --help")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
