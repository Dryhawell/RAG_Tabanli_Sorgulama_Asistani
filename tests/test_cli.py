from rag.cli import build_parser


def test_cli_parser_subcommands():
    parser = build_parser()
    args = parser.parse_args(["list"])
    assert args.command == "list"
    args = parser.parse_args(["rebuild", "--embedding", "mini-multi"])
    assert args.command == "rebuild"
    assert args.embedding == "mini-multi"
    args = parser.parse_args(
        ["ingest", "a.pdf", "--from-data", "--folder", "hukuk", "--tags", "a,b"]
    )
    assert args.paths == ["a.pdf"]
    assert args.from_data is True
    assert args.folder == "hukuk"
    assert args.tags == "a,b"
    args = parser.parse_args(["eval", "--embedding", "hash"])
    assert args.command == "eval"
    assert args.embedding == "hash"
    args = parser.parse_args(["judge", "--mode", "heuristic"])
    assert args.command == "judge"
    assert args.mode == "heuristic"
    args = parser.parse_args(["stats", "--json"])
    assert args.command == "stats"
    assert args.json is True
    args = parser.parse_args(["prometheus", "--dump"])
    assert args.command == "prometheus"
    assert args.dump is True
