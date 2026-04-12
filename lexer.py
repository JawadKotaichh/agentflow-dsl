from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    AGENT = auto()
    TOOL = auto()
    TASK = auto()
    ACTION = auto()
    SYSTEM = auto()
    RUN = auto()
    IF = auto()
    FOR = auto()
    IN = auto()
    STRING_KEYWORD = auto()
    INT_KEYWORD = auto()
    BOOL_KEYWORD = auto()
    LIST_KEYWORD = auto()

    ID = auto()
    STRING_LIT = auto()
    INT_LIT = auto()
    BOOL_LIT = auto()

    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACKET = auto()
    RIGHT_BRACKET = auto()
    COMMA = auto()
    COLON = auto()
    DOT = auto()
    ASSIGN = auto()
    NOT_EQUAL = auto()
    EQUAL_EQUAL = auto()
    PLUS = auto()
    MINUS = auto()
    MULT = auto()
    DIV = auto()
    GREATER_EQUAL = auto()
    GREATER = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    ARROW = auto()

    EOF = auto()


@dataclass
class Token:
    type: TokenType
    lexeme: str
    line: int
    column: int

    def __repr__(self) -> str:
        return (
            f"Token(type={self.type.name}, lexeme={self.lexeme!r}, "
            f"line={self.line}, column={self.column})"
        )


class LexerError(Exception):
    pass


class Lexer:
    KEYWORDS = {
        "agent": TokenType.AGENT,
        "tool": TokenType.TOOL,
        "task": TokenType.TASK,
        "action": TokenType.ACTION,
        "system": TokenType.SYSTEM,
        "run": TokenType.RUN,
        "if": TokenType.IF,
        "for": TokenType.FOR,
        "in": TokenType.IN,
        "string": TokenType.STRING_KEYWORD,
        "int": TokenType.INT_KEYWORD,
        "bool": TokenType.BOOL_KEYWORD,
        "list": TokenType.LIST_KEYWORD,
    }

    BOOL_LITERALS = {
        "true": TokenType.BOOL_LIT,
        "false": TokenType.BOOL_LIT,
    }

    def __init__(self, source: str):
        self.source = source
        self.length = len(source)
        self.pos = 0
        self.line = 1
        self.column = 1

    def is_empty(self):
        return self.pos >= self.length

    def skip_whitespace_and_comments(self):
        while not self.is_empty():
            ch = self.peek()

            if ch in " \t\r":
                self.advance()
                continue

            if ch == "\n":
                self.advance()
                continue
            break

    def read_identifier_or_keyword(self):
        start = self.pos
        start_line = self.line
        start_col = self.column

        while not self.is_empty() and (self.peek().isalnum() or self.peek() == "_"):
            self.advance()

        lexeme = self.source[start : self.pos]

        if lexeme in self.BOOL_LITERALS:
            return Token(TokenType.BOOL_LIT, lexeme, start_line, start_col)

        token_type = self.KEYWORDS.get(lexeme, TokenType.ID)
        return Token(token_type, lexeme, start_line, start_col)

    def read_integer(self):
        start = self.pos
        start_line = self.line
        start_col = self.column

        while not self.is_empty() and self.peek().isdigit():
            self.advance()

        lexeme = self.source[start : self.pos]
        return Token(TokenType.INT_LIT, lexeme, start_line, start_col)

    def read_string(self):
        start_line = self.line
        start_col = self.column

        self.advance()  # consume the openning "
        chars = []

        while not self.is_empty() and self.peek() != '"':
            if self.peek() == "\n":
                raise LexerError(
                    f"Unterminated string at line {start_line}, column {start_col}"
                )

            if self.peek() == "\\":
                self.advance()
                if self.is_empty():
                    raise LexerError(
                        f"Unterminated escape sequence at line {self.line}, column {self.column}"
                    )

                esc = self.peek()
                escape_map = {
                    "n": "\n",
                    "t": "\t",
                    '"': '"',
                    "\\": "\\",
                }
                chars.append(escape_map.get(esc, esc))
                self.advance()
            else:
                chars.append(self.peek())
                self.advance()

        if self.is_empty():
            raise LexerError(
                f"Unterminated string at line {start_line}, column {start_col}"
            )

        self.advance()  # consume closing "
        value = "".join(chars)
        return Token(TokenType.STRING_LIT, value, start_line, start_col)

    def make_token(
        self,
        token_type: TokenType,
        lexeme: str,
        line: int,
        column: int,
    ) -> Token:
        return Token(token_type, lexeme, line, column)

    def peek(self) -> str:
        if self.is_empty():
            return "\0"
        return self.source[self.pos]

    def peek_next(self) -> str:
        if self.pos + 1 >= self.length:
            return "\0"
        return self.source[self.pos + 1]

    def advance(self):
        ch = self.source[self.pos]
        self.pos += 1

        if ch == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1

        return ch

    def tokenize(self):
        tokens = []

        while not self.is_empty():
            self.skip_whitespace_and_comments()
            if self.is_empty():
                break

            start_line = self.line
            start_col = self.column
            ch = self.peek()

            if ch == '"':
                tokens.append(self.read_string())
                continue

            if ch.isdigit():
                tokens.append(self.read_integer())
                continue

            if ch.isalpha() or ch == "_":
                tokens.append(self.read_identifier_or_keyword())
                continue

            two = self.source[self.pos : self.pos + 2]

            if two == "->":
                tokens.append(
                    self.make_token(TokenType.ARROW, "->", start_line, start_col)
                )
                self.advance()
                self.advance()
                continue
            if two == "!=":
                tokens.append(
                    self.make_token(TokenType.NOT_EQUAL, "!=", start_line, start_col)
                )
                self.advance()
                self.advance()
                continue
            if two == "==":
                tokens.append(
                    self.make_token(TokenType.EQUAL_EQUAL, "==", start_line, start_col)
                )
                self.advance()
                self.advance()
                continue
            if two == ">=":
                tokens.append(
                    self.make_token(
                        TokenType.GREATER_EQUAL, ">=", start_line, start_col
                    )
                )
                self.advance()
                self.advance()
                continue
            if two == "<=":
                tokens.append(
                    self.make_token(TokenType.LESS_EQUAL, "<=", start_line, start_col)
                )
                self.advance()
                self.advance()
                continue

            single_char_tokens = {
                "{": TokenType.LEFT_BRACE,
                "}": TokenType.RIGHT_BRACE,
                "(": TokenType.LEFT_PAREN,
                ")": TokenType.RIGHT_PAREN,
                "[": TokenType.LEFT_BRACKET,
                "]": TokenType.RIGHT_BRACKET,
                ",": TokenType.COMMA,
                ":": TokenType.COLON,
                ".": TokenType.DOT,
                "=": TokenType.ASSIGN,
                "+": TokenType.PLUS,
                "-": TokenType.MINUS,
                "*": TokenType.MULT,
                "/": TokenType.DIV,
                ">": TokenType.GREATER,
                "<": TokenType.LESS,
            }

            if ch in single_char_tokens:
                token_type = single_char_tokens[ch]
                tokens.append(self.make_token(token_type, ch, start_line, start_col))
                self.advance()
                continue

            raise LexerError(
                f"Unexpected character {ch!r} at line {self.line}, column {self.column}"
            )

        tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        return tokens


if __name__ == "__main__":
    source = """
        agent Researcher {
            tool web_search
            tool llm
            task gather(string topic) -> string data {
                action: web_search(topic)
                action: llm("summarize results")
            }
        }
        agent Analyzer {
            tool llm
            task sentiment(string text) -> string result {
                action: llm("detect sentiment")
            }
        }
        system {
            list topics = ["AI","Robotics","Security"]
            int i = 0
            bool negative_found = false
            for t in topics {
                string data = run Researcher.gather(t)
                string sentiment = run Analyzer.sentiment(data)
                if sentiment == "negative" {
                    negative_found = true
                }
                i = i + 1
            }
        }
    """
    lexer = Lexer(source)
    try:
        tokens = lexer.tokenize()
        for token in tokens:
            print(token)
    except LexerError as e:
        print("Lexer error:", e)
