from rag.cli import build_parser


def test_cli_parser_subcommands():
    parser = build_parser()
    args = parser.parse_args(["list"])
    assert args.command == "list"
    args = parser.parse_args(["rebuild", "--embedding", "mini-multi"])
    assert args.command == "rebuild"
    assert args.embedding == "mini-multi"
    args = parser.parse_args(["ingest", "a.pdf", "--from-data"])
    assert args.paths == ["a.pdf"]
    assert args.from_data is True
